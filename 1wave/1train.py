############################################################
#                       波动方程                            #
#                    step1:用于训练模型                     #
############################################################

from pathlib import Path
import sys

# 将项目根目录（上一层）加入导入路径
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
    
    # 检查数据是否正确
    # print(f"x'feat type:{type(X_feat)},y'feat type:{type(Y_feat)},edges'feat type:{type(edge_index)},")
    # print(f"x'feat shape:{X_feat.shape},y'feat shape:{Y_feat.shape},edges'feat shape:{edge_index.shape},")
    # print(f"x feature:{X_feat[1][128*64+64]} \ny feature:{Y_feat[1][128*64+64]}")
    # print(X_feat.dtype)
    # edge_index = edge_index.double() 
    # print(edge_index.dtype)

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

    # 设置训练参数
    init_lr = 1e-3
    opt = torch.optim.Adam(pgn.parameters(), lr=init_lr, weight_decay=1e-6)
    # total_epochs = 1000
    total_epochs = 200  # 用于测试，轮次较低
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

                # ----------------- 注入噪声对抗噪声 -----------------
                if noise_level > 0:
                    ginput.x = tool.inject_noise(ginput.x, noise_level=noise_level)

                opt.zero_grad()

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

        train_loss = total_loss/num_items
        # cur_loss = total_loss/num_items
        print(f"\nloss:{train_loss}")

        ###################################################
        pgn.eval()
        val_total_loss = 0.0
        val_items = 0

        with torch.no_grad(): # 验证时不计算梯度，省显存
            for val_data in newtestloader:
                val_data = val_data.cuda()
                # 注意：验证集数据保持原样，不要注入噪声！我们要看模型对真实数据的表现。
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

            # 1. 提取符号回归所需的消息 (Messages)
            cur_msgs = tool.get_messages(pgn, loss_type, msg_dim, newtestloader, dim=2)
            cur_msgs['loss'] = cur_val_loss
            messages_best = cur_msgs
            
            # 2. ！！！立即保存模型权重到硬盘！！！
            # 这样即使后面 900 轮过拟合了，硬盘里存的也是这一刻最好的
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
