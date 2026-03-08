############################################################
#                       Wave Equation                      #
#                    Step 1: For Model Training            #
############################################################

import model
import numpy as np
from model import PGN
from torch_geometric.data import Data, DataLoader
import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pickle as pkl
import config
import tool



if __name__ == "__main__":
    # Load dataset
    save_path=config.source_path
    with np.load(save_path, allow_pickle=True) as data:
        values = data['solution']
        t = data['t_eval']
        metadata = data['parameters'].item()
        # X_feat, Y_feat = tool.bulid_feat(data) #6,1
        X_feat, Y_feat = tool.build_feat_SG(data,7,2) #6,1
    
    estimated_noise_level = tool.estimate_noise_level(values)
    print("####################################\n")
    print(f"Estimated noise level: {estimated_noise_level:.4f}")

    if estimated_noise_level < 0.005:
        noise_level = estimated_noise_level * 0.0
    elif estimated_noise_level < 0.02:
        noise_level = estimated_noise_level * 0.4
    elif estimated_noise_level < 0.08:
        noise_level = estimated_noise_level * 0.7
    elif estimated_noise_level < 0.15:
        noise_level = estimated_noise_level * 0.2   
    else:
        noise_level = 0.03
    # Keep three decimal places
    noise_level = round(noise_level, 3)
    print(f"Using noise level: {noise_level}")
    
    # Grid granularity
    N_x = metadata['Nx']
    N_y = metadata['Ny']
    n = N_x * N_y
    X_feat=torch.from_numpy(X_feat)
    Y_feat=torch.from_numpy(Y_feat)
    edge_index = model.get_edge_index(Ny=N_y,Nx=N_x)
    
    # Check if data is correct
    # print(f"x'feat type:{type(X_feat)},y'feat type:{type(Y_feat)},edges'feat type:{type(edge_index)},")
    # print(f"x'feat shape:{X_feat.shape},y'feat shape:{Y_feat.shape},edges'feat shape:{edge_index.shape},")
    # print(f"x feature:{X_feat[1][128*64+64]} \ny feature:{Y_feat[1][128*64+64]}")
    # print(X_feat.dtype)
    # edge_index = edge_index.double() 
    # print(edge_index.dtype)

    # Set model parameters
    aggr = config.aggr
    hidden = config.hidden
    msg_dim = config.msg_dim
    dim=config.dim
    out_dim=config.out_dim
    loss_type = '_l1_'  # Loss type
    n_f = len(X_feat[0][0]) # Feature dimension
    print(f"Input feature dimension:{n_f}")

    pgn = PGN(n_f, msg_dim, dim, out_dim, hidden=hidden, edge_index=edge_index, aggr=aggr).cuda()
    messages_over_time = []
    pgn = pgn.cuda()

    # Perform one iteration to check if it runs properly
    # _q = Data(
    # x=X_feat[0].cuda(),
    # edge_index=edge_index.cuda(),
    # y=Y_feat[0].cuda())
    # batch=1
    # print(pgn(_q.x, _q.edge_index))
    # print(pgn.just_derivative(_q).shape)
    # print(_q.y.shape)
    # print(new_loss(pgn,_q,loss_type))
    
    batch = 3200
    trainloader = DataLoader(
        [Data(
            Variable(X_feat[i]),
            edge_index=edge_index,
            y=Variable(Y_feat[i])) for i in range(len(Y_feat))],
        batch_size=batch,
        shuffle=True
    )

    # Set training parameters
    init_lr = 1e-3
    opt = torch.optim.Adam(pgn.parameters(), lr=init_lr, weight_decay=1e-6)
    # total_epochs = 1000
    total_epochs = 200  # Used for testing, lower number of epochs
    batch_per_epoch = int(10000 / (batch/32.0))

    sched = CosineAnnealingLR(
        opt,                                        
        T_max=total_epochs * batch_per_epoch,     # Total steps: epochs * steps_per_epoch (corresponds to original total training steps)
        eta_min=init_lr * 1e-2
    )
    epoch = 0

    # Intermediate state recording
    np.random.seed(42)
    test_idxes = np.random.randint(0, len(X_feat), 100)

    newtestloader = DataLoader(
    [Data(
        X_feat[i],
        edge_index=edge_index,
        y=Y_feat[i]) for i in test_idxes],
        batch_size=len(Y_feat),
    shuffle=False)

    recorded_models = []

    # Start training
    print("Start training")
    best_val_loss = float('inf')
    messages_best = None

    for epoch in tqdm(range(epoch, total_epochs)):
        pgn.cuda()
        pgn.train()

        total_loss = 0.0
        i = 0
        num_items = 0

        while i < batch_per_epoch:
            for ginput in trainloader:
                if i >= batch_per_epoch:
                    break
                ginput.x = ginput.x.cuda()
                ginput.y = ginput.y.cuda()
                ginput.edge_index = ginput.edge_index.cuda()
                ginput.batch = ginput.batch.cuda()

                # ----------------- Inject noise to combat noise -----------------
                if noise_level > 0:
                    ginput.x = tool.inject_noise(ginput.x, noise_level=noise_level)

                opt.zero_grad()

                # Select different loss based on loss type
                if loss_type in ['_l1_', '_kl_']:
                    loss, reg = tool.new_loss(pgn, ginput, loss_type, batch, n)
                    ((loss + reg)/int(ginput.batch[-1]+1)).backward()
                else:
                    loss = pgn.loss(ginput, square=False)
                    (loss/int(ginput.batch[-1]+1)).backward()
                opt.step()
                sched.step()

                total_loss += loss.item()
                i += 1
                num_items += int(ginput.batch[-1]+1)

        train_loss = total_loss/num_items
        # cur_loss = total_loss/num_items
        print(f"\nloss:{train_loss}")

        ###################################################
        pgn.eval()
        val_total_loss = 0.0
        val_items = 0

        with torch.no_grad(): # Do not calculate gradients during validation to save VRAM
            for val_data in newtestloader:
                val_data = val_data.cuda()
                # Note: Keep validation set data as is, do not inject noise! We want to see the model's performance on real data.
                # v_loss = pgn.loss(val_data, square=False) 
                if loss_type in ['_l1_', '_kl_']:
                    v_loss, _ = tool.new_loss(pgn, val_data, loss_type, batch, n)
                else:
                    v_loss = pgn.loss(val_data, square=False)

                val_total_loss += v_loss.item()
                val_items += int(val_data.batch[-1]+1)  
        
        cur_val_loss = val_total_loss / val_items

        if cur_val_loss < best_val_loss:
            best_val_loss = cur_val_loss

            # 1. Extract messages required for PySR symbolic regression (Messages)
            cur_msgs = tool.get_messages(pgn, loss_type, msg_dim, newtestloader, dim=2)
            cur_msgs['loss'] = cur_val_loss
            messages_best = cur_msgs
            
            # 2. !!! Immediately save model weights to disk !!!
            # This way, even if it overfits in the subsequent epochs, the best one at this moment is saved on disk
            torch.save(pgn.state_dict(), f'result/models_best{config.name}.pth')
            # print(f"  --> Model Saved at epoch {epoch}")
            
        pgn.cpu()
        recorded_models.append(pgn.state_dict())

    print(f"Training finished. Best validation Loss: {best_val_loss:.6f}")
    
    # ------------------- Save results to disk -------------------
    print("Saving message data...")
    if messages_best is not None:
        t_columns = [col for col in messages_best.columns if col.startswith('t') and col != 't']
        if len(t_columns) > 0:
            keep_t = t_columns[0]
            drop_t = t_columns[1:]
            # Delete multiple t columns, keep only one t
            messages_best = messages_best.drop(columns=drop_t)
            messages_best = messages_best.rename(columns={keep_t: 't'})

        # Save message history list (for PySR symbolic regression)
        pkl.dump(messages_best, open(f'result/messages_best{config.name}.pkl', 'wb'))
    else:
        print("Warning: No better model found, the model might not have converged or the Loss kept increasing.")

    # Note: models_best.pth has already been saved inside the loop, no need to save again here
    print(f"Best model saved at: result/models_best{config.name}.pth")