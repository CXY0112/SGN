############################################################
#                       波动方程                            #
#             step4:构造图求解器，进行测试                   #
############################################################


import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import  MessagePassing
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 将项目根目录（上一层）加入导入路径
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import model
import config
import tool


class CustomMsgFn(nn.Module):
    def __init__(self):
        super(CustomMsgFn, self).__init__()

    def forward(self, tmp):
        # x_i: [batch_edges, 3], x_j: [batch_edges, 3]
        # 拼接输入：cat([x_i, x_j], dim=-1) -> [batch_edges, 6]
        input_cat = tmp
        
        # 解包变量（按公式）
        # msg_feature_names = [x1, y1, t1, ux1, uy1, u1(x, y, t-1)],[x2, y2, t2, ux2, uy2, u2(x, y, t-1)]
        x1, y1, t, ux1, uy1, u_prev1, x2, y2, t2, ux2, uy2, u_prev2 = input_cat[:, 0], input_cat[:, 1], input_cat[:, 2], \
                                 input_cat[:, 3], input_cat[:, 4], input_cat[:, 5], input_cat[:, 6], input_cat[:, 7], \
                                 input_cat[:, 8], input_cat[:, 9], input_cat[:, 10], input_cat[:, 11]
        # dx=x1-x2
        # dy=y1-y2
        # 计算 msg1 = sin(x1 + x2) + x3 * y3
        msg1 = (u_prev1 + (-torch.sin(u_prev1) - 2.8246412)*torch.sin(u_prev2))/(-6.849007)
        # 输出 [batch_edges, 2]
        msg = torch.stack([msg1], dim=-1)
        return msg

class CustomNodeFn(nn.Module):
    def __init__(self, output_dim=1):
        super(CustomNodeFn, self).__init__()
        self.output_dim=output_dim

    def forward(self, aggr_out):
        # aggr_out: [n, msg_dim=2]，聚合消息
        # node_feature_names = [x, y, t, ux, uy, u(x, y, t-1)],['e']
        # 解包变量
        x, y, t, ux, uy, u_prev, e = aggr_out[:, 0], aggr_out[:, 1], aggr_out[:, 2], aggr_out[:, 3], aggr_out[:, 4], aggr_out[:, 5], aggr_out[:, 6]  
        
        # 具体公式
        # out = (-0.0049376567 + (torch.sin(y * 3.0965688) * torch.sin(x * 3.084605))) * torch.cos(t / -0.31685314)
        out = u_prev - torch.sin((torch.sin(t * -3.1070971) * torch.sin(y * -3.1335773)) * 0.038306106)
        if self.output_dim == 1:
            out = out.unsqueeze(-1)
        return out

class InterpretableGN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, edge_index, aggr=config.aggr):
        super(InterpretableGN, self).__init__(aggr=aggr)  
        # 显示公式定义的生成消息函数
        self.msg_fnc = CustomMsgFn()
        # 显示公式定义的消息聚合函数
        self.node_fnc = CustomNodeFn(1)
        self.edge_index = edge_index
        self.ndim = ndim
    
    def forward(self, x, edge_index):
        # x is [n, n_f],节点特征张量，其中 n 是节点数，n_f 是每个节点的特征维度
        # edge_index：边索引张量，形状为 (2, num_edges)
        x = x # ？

        # propagate调用了message()，又根据aggr进行了聚合，最后调用了update()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
      
    def message(self, x_i, x_j):
        # x_i has shape [n_e, n_f]; x_j has shape [n_e, n_f]
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        tmp = tmp.float()
        return self.msg_fnc(tmp)
    
    def update(self, aggr_out, x=None):
        # aggr_out has shape [n, msg_dim]
        tmp = torch.cat([x, aggr_out], dim=1)
        tmp = tmp.float()
        return self.node_fnc(tmp) #[n, nupdate]

# 加载数据集
save_path="data/wave_solution_2d(32).npz"
# save_path=config.source_path
with np.load(save_path, allow_pickle=True) as data:
    values = data['solution']
    data_x = data['x']
    data_y = data['y']
    t = data['t_eval']
    metadata = data['parameters'].item()
    X_feat, Y_feat = tool.build_feat_SG(data)
# 网格粒度
N_x = metadata['Nx']
N_y = metadata['Ny']
n = N_x * N_y
X_feat=torch.from_numpy(X_feat)
Y_feat=torch.from_numpy(Y_feat)
edge_index = model.get_edge_index(Ny=N_y,Nx=N_x)
aggr = config.aggr
hidden = config.hidden
msg_dim = config.msg_dim  #消息维度
dim=config.dim
n_f = len(X_feat[0][0]) #  特征维度
print(f"nf={n_f}")

pgn = InterpretableGN(n_f, msg_dim, dim, edge_index=edge_index, aggr=aggr).cuda()

# 设置为评估模式
pgn.eval()

res_t=[]
prev_pred = None  # 用于存储前一步预测值
for i in range(len(X_feat)):
    if i == 0:
        # 第一个时间步：使用真实的初始 u_prev
        _input_feat = X_feat[i].clone()  # (Nx*Ny, 4)
    else:
        # 后续时间步：使用前一步的预测值替换第四维
        xy_t_current = X_feat[i][:, :5].clone()  # 前三维：x, y, t (Nx*Ny, 3)

        _input_feat = torch.cat([xy_t_current, prev_pred], dim=1)  # 拼接预测 u_prev (Nx*Ny, 4)
    
    _q = Data(
        x=_input_feat.cuda(),
        edge_index=edge_index.cuda()
    )
    res = pgn(_q.x, _q.edge_index)  # 预测当前 u (Nx*Ny,)
    res_t.append(res.cpu())  # 移到CPU存储，避免GPU内存积累
    prev_pred = res.cpu()  # 更新前一步预测值（用于下轮）

# 定义保存路径
prediction_save_path = f"result/pgn_prediction_2d{config.name}_final.npz"

u_pred_tensors = torch.stack(res_t).cpu()
u_pred_flat = u_pred_tensors.detach().numpy()
u_solution_pred = u_pred_flat.reshape(len(t)-1, N_y, N_x)

result = {
    'solution': u_solution_pred,
    'x': data_x,
    'y': data_y,
    't_eval': t-1, 
    'parameters': metadata 
}

np.savez(prediction_save_path, **result)
print(f"结果已保存至 {prediction_save_path}")

# 计算误差：真实解与预测解形状对齐
u_true = values[1:]  # 从 t=1 开始，与预测对应 (Nt, Ny, Nx)
assert u_true.shape == u_solution_pred.shape, f"Shape mismatch: {u_true.shape} vs {u_solution_pred.shape}"

# ===== 误差计算部分 =====
mse = np.mean((u_solution_pred - u_true) ** 2)

# 按时间步统计误差趋势（可视化分析用）
mse_t = np.mean((u_solution_pred - u_true) ** 2, axis=(1, 2))  # (Nt,)

print(f"全局 MSE: {mse:.6e}")
# print(f"平均每步 MSE: {np.mean(mse_t):.6e} ± {np.std(mse_t):.6e}")

# ===== 保存预测结果与误差 =====
result = {
    'solution': u_solution_pred,
    'x': data_x,
    'y': data_y,
    't_eval': t[1:],  # 注意此处应对齐预测步
    'parameters': metadata,
    'mse': mse,
    'mse_t': mse_t,
}

plt.figure(figsize=(6,4))
plt.plot(t[1:], mse_t, label='MSE over time')
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()
plt.title('Prediction Error Evolution')
plt.grid(True)
plt.tight_layout()
plt.show()