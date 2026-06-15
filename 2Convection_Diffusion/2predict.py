############################################################
#                  Convection-Diffusion                    #
#                 step2:用GNN对数据拟合                     #
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
from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt
import config
import tool

# 加载数据集
save_path=config.source_path
with np.load(save_path, allow_pickle=True) as data:
    values = data['solution']
    data_x = data['x']
    data_y = data['y']
    t = data['t_eval']
    metadata = data['parameters'].item()
    X_feat, Y_feat = tool.build_feat_SG_2(data,7,2)
    
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
out_dim=config.out_dim
n_f = len(X_feat[0][0]) #  特征维度
print(f"nf={n_f}")

pgn = PGN(n_f, msg_dim, dim, out_dim, hidden=hidden, edge_index=edge_index, aggr=aggr).cuda()
pgn.load_state_dict(torch.load(f'result/models_best{config.name}.pth',map_location='cuda'))

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
        xy_t_current = X_feat[i][:, :7].clone()  # 前三维：x, y, t (Nx*Ny, 3)

        _input_feat = torch.cat([xy_t_current, prev_pred], dim=1)  # 拼接预测 u_prev (Nx*Ny, 4)
    
    _q = Data(
        x=_input_feat.cuda(),
        edge_index=edge_index.cuda()
    )
    res = pgn(_q.x, _q.edge_index)  # 预测当前 u (Nx*Ny,)
    res_t.append(res.cpu())  # 移到CPU存储，避免GPU内存积累
    prev_pred = res.cpu()  # 更新前一步预测值（用于下轮）

# 定义保存路径
prediction_save_path = f"result/pgn_prediction_2d{config.name}.npz"

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
