############################################################
#                     NavierStokes方程                     #
#                    step1:用于训练模型                     #
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
    # 加载数据集
    save_path=config.source_path
    with np.load(save_path, allow_pickle=True) as data:
        values = data['x_solution']     #取x方向速度场简单测试
        metadata = data['parameters'].item()
        X_feat, Y_feat = tool.build_feat_2_SG(data) #9,2

    estimated_noise_level = tool.estimate_noise_level(values,7,2)
    print("####################################\n")
    print(f"Estimated noise level: {estimated_noise_level:.4f}")

    if estimated_noise_level < 0.005:
        noise_level = estimated_noise_level * 0.5
    elif estimated_noise_level < 0.02:
        noise_level = estimated_noise_level * 0.4
    elif estimated_noise_level < 0.08:
        noise_level = estimated_noise_level * 0.7
    elif estimated_noise_level < 0.15:
        noise_level = estimated_noise_level * 0.2   
    else:
        noise_level = 0.03

    # 保留三位小数
    noise_level = round(noise_level, 3)
    print(f"Using noise level: {noise_level}")

    # 网格粒度
    N_x = metadata['Nx']
    N_y = metadata['Ny']
    n = N_x * N_y
    X_feat=torch.from_numpy(X_feat)
    Y_feat=torch.from_numpy(Y_feat)
    edge_index = model.get_edge_index(Ny=N_y,Nx=N_x)
    
    # 设置模型参数
    aggr = config.aggr
    hidden = config.hidden
    msg_dim = config.msg_dim
    dim=config.dim
    out_dim=config.out_dim
    loss_type = '_l1_'  # 损失类型
    n_f = len(X_feat[0][0]) #  特征维度
    print(f"Input feature dimension:{n_f}")

    pgn = PGN(n_f, msg_dim, dim, out_dim, hidden=hidden, edge_index=edge_index, aggr=aggr).cuda()
    messages_over_time = []
    pgn = pgn.cuda()

    # 进行一次迭代，检查是否正常运行
    # _q = Data(
    # x=X_feat[0].cuda(),
    # edge_index=edge_index.cuda(),
    # y=Y_feat[0].cuda())
    # batch=1
    # print(pgn(_q.x, _q.edge_index))
    # print(pgn.just_derivative(_q).shape)
    # print(_q.y.shape)
    # print(tool.new_loss(pgn,_q,loss_type,batch,n))
    
    batch = 3200
    trainloader = DataLoader(
        [Data(
            Variable(X_feat[i]),
            edge_index=edge_index,
            y=Variable(Y_feat[i])) for i in range(len(Y_feat))],
        batch_size=batch,
        shuffle=True
    )

    # 设置训练参数
    init_lr = 1e-3
    opt = torch.optim.Adam(pgn.parameters(), lr=init_lr, weight_decay=1e-6)
    # total_epochs = 1000
    total_epochs = 300  
    batch_per_epoch = int(10000 / (batch/32.0))

    sched = CosineAnnealingLR(
        opt,                                    
        T_max=total_epochs * batch_per_epoch,     # 总步数：epochs * steps_per_epoch（对应原总训练步数）
        eta_min=init_lr * 1e-2                   
    )
    epoch = 0

    # 中间状态记录
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

    # 开始训练
    print("开始训练")
    best_val_loss = float('inf')
    messages_best = None
    for epoch in tqdm(range(epoch, total_epochs)):
        pgn.cuda()
        total_loss = 0.0
        i = 0
        num_items = 0
        while i < batch_per_epoch:
            for ginput in trainloader:
                if i >= batch_per_epoch:
                    break
                opt.zero_grad()
                ginput.x = ginput.x.cuda()
                ginput.y = ginput.y.cuda()
                ginput.edge_index = ginput.edge_index.cuda()
                ginput.batch = ginput.batch.cuda()

                # ----------------- 注入噪声对抗噪声 -----------------
                if noise_level > 0:
                    ginput.x = tool.inject_noise(ginput.x, noise_level=noise_level)

                # 根据loss类型的不同，选择不同的损失
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

        cur_loss = total_loss/num_items
        print(f"\nloss:{cur_loss}")

        ###################################################
        pgn.eval()
        val_total_loss = 0.0
        val_items = 0

        with torch.no_grad(): # 验证时不计算梯度，省显存
            for val_data in newtestloader:
                val_data = val_data.cuda()
                if loss_type in ['_l1_', '_kl_']:
                    v_loss, _ = tool.new_loss(pgn, val_data, loss_type, batch, n)
                else:
                    v_loss = pgn.loss(val_data, square=False)

                val_total_loss += v_loss.item()
                val_items += int(val_data.batch[-1]+1)  
        
        cur_val_loss = val_total_loss / val_items

        if cur_val_loss < best_val_loss:
            best_val_loss = cur_val_loss

            # 1. 提取符号回归所需的消息 (Messages)
            cur_msgs = tool.get_messages(pgn, loss_type, msg_dim, newtestloader, dim=2)
            cur_msgs['loss'] = cur_val_loss
            messages_best = cur_msgs
            
            # 2. ！！！立即保存模型权重到硬盘！！！
            torch.save(pgn.state_dict(), f'result/models_best{config.name}.pth')
            # print(f"  --> Model Saved at epoch {epoch}")
            
        pgn.cpu()
        recorded_models.append(pgn.state_dict())

    print(f"训练结束。最佳验证集 Loss: {best_val_loss:.6f}")
    
    # ------------------- 结果落盘 -------------------
    print("正在保存消息数据...")
    if messages_best is not None:
        t_columns = [col for col in messages_best.columns if col.startswith('t') and col != 't']
        if len(t_columns) > 0:
            keep_t = t_columns[0]
            drop_t = t_columns[1:]
            # 删除多个 t 列，只保留一个 t
            messages_best = messages_best.drop(columns=drop_t)
            messages_best = messages_best.rename(columns={keep_t: 't'})

        # 保存消息历史列表 (用于 PySR 符号回归)
        pkl.dump(messages_best, open(f'result/messages_best{config.name}.pkl', 'wb'))
    else:
        print("警告: 未找到更优模型，可能模型未收敛或 Loss 持续上升。")

    # 注意：models_best.pth 已经在循环内部保存过了，这里不需要再次保存
    print(f"最佳模型已保存在: result/models_best{config.name}.pth")

    #     cur_msgs = tool.get_messages(pgn,loss_type,msg_dim,newtestloader,dim=2)
    #     cur_msgs['loss'] = cur_loss
    #     if cur_loss < best_loss:
    #         best_loss = cur_loss
    #         messages_best = cur_msgs
        
    #     pgn.cpu()
    #     recorded_models.append(pgn.state_dict())

    # print("训练完成，正在保存数据")
    # t_columns = [col for col in messages_best.columns if col.startswith('t') and col != 't']
    # keep_t = t_columns[0]
    # drop_t = t_columns[1:]
    # # 删除多个t列，只保留一个t
    # messages_best = messages_best.drop(columns=drop_t)
    # messages_best = messages_best.rename(columns={keep_t: 't'})

    # # 保存训练数据，消息历史列表
    # print("正在保存消息...")
    # # 保存训练数据，消息历史列表
    # pkl.dump(messages_best,open(f'result/messages_best{config.name}.pkl', 'wb'))

    # print("正在保存模型...")
    # torch.save(pgn.state_dict(), f'result/models_best{config.name}.pth')