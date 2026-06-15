############################################################
#                       波动方程                            #
#                 step3:用pySR提取公式                      #
############################################################

from pysr import PySRRegressor
import pickle as pkl
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 将项目根目录（上一层）加入导入路径
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from scipy.signal import savgol_filter

def bulid_aggr_feat(result,data_node):
    """
    data_node的特征为[x,y,t,e1,e2,e3,e4]
    我需要从result中构造出[u,x,y,t,ux, uy, u(x, y, t-1)]
    然后将x,y,t相等的项合并为feat[u,x,y,t,e1,e2,e3,e4,ux, uy, u(x, y, t-1)]
    注意：以data_node中x,y,t为主，即len(feat)=len(data_node)
    feat: (len(data_node), 7+4) -> [u(x, y, t),x, y, t, ux, uy, u(x, y, t-1),e1,e2,e3,e4]  # 排除初始步，只取从 i=1 开始
    """
    u = result['solution']    # (Nt+1, Ny, Nx)
    x_arr = result['x']       # (Nx,)
    y_arr = result['y']       # (Ny,)
    t_arr = result['t_eval']  # (Nt+1,)

    original_Nt_plus1, Ny, Nx = u.shape
    Nt = original_Nt_plus1 - 1
    NxNy = Nx * Ny
    dx = x_arr[1] - x_arr[0]
    dy = y_arr[1] - y_arr[0]

    # 生成空间坐标网格
    X, Y = np.meshgrid(x_arr, y_arr, indexing='xy')  # (Ny, Nx)

    # 初始化容器
    feat_list = []
    
    # 遍历时间步，从 i=1 开始
    for i in range(1, original_Nt_plus1):
        t = t_arr[i]  
        # 每个时间步的空间特征 [x, y, t]
        # xy_t = np.stack([X.flatten(), Y.flatten(), np.full(X.size, t)], axis=1)  # (Nx*Ny, 3)
        
        # 上一个时间步的 u 值：u[i-1]，保持 2D 用于梯度计算
        u_prev_2d = u[i-1]  # (Ny, Nx)
        # 计算一阶偏导 ux (∂u/∂x) 和 uy (∂u/∂y)
        # ux_2d = np.gradient(u_prev_2d, x_arr, axis=1)  # (Ny, Nx)，沿 x 方向
        # uy_2d = np.gradient(u_prev_2d, y_arr, axis=0)  # (Ny, Nx)，沿 y 方向
        ux_2d = savgol_filter(u_prev_2d, 7, 2,deriv=1,delta=dx,axis=1)  # 平滑处理
        uy_2d = savgol_filter(u_prev_2d, 7, 2,deriv=1,delta=dy,axis=0)  # 平滑处理
        u_temp = savgol_filter(u_prev_2d, 7, 2, deriv=0, axis=1)
        u_prev_2d_smooth = savgol_filter(u_temp, 7, 2, deriv=0, axis=0)
        prev_u = u_prev_2d_smooth.flatten() # (Nx*Ny,) - 使用平滑后的 u
        feat_step = pd.DataFrame({
            'x': X.flatten(),
            'y': Y.flatten(),
            't': np.full(X.size, np.round(t, 6)),  # 时间取当前步
            'u': u[i].flatten(),
            'ux': ux_2d.flatten(),
            'uy': uy_2d.flatten(),
            # 'u_prev': u_prev_2d.flatten()
            'u_prev': prev_u
        })
        feat_list.append(feat_step)
    # 拼接所有时间步
    feat_df = pd.concat(feat_list, ignore_index=True)
    # 对浮点数进行标准化取整（防止精度误差）
    feat_df = feat_df.round(6)
    data_node = data_node.round(6)

    print(feat_df.head())
    print("feat_df shape:", feat_df.shape)
    print(data_node.head())
    print("data_node shape:", data_node.shape)

    # 合并 PDE 特征和 message 特征
    merged = pd.merge(
        data_node,
        feat_df,
        on=['x', 'y', 't'],
        how='left'
    )

    # 重新排序列
    merged = merged[['u', 'x', 'y', 't', 'ux', 'uy', 'u_prev', 'e1', 'e2', 'e3', 'e4']]
    return merged

# 步骤 1: 加载保存的数据
best_message =pkl.load(open(f'result/messages_best{config.name}.pkl', 'rb'))
print("Loaded DataFrame shape:", best_message.shape)
print(best_message.head()) #查看完整的特征

# 通过最大标准差获取相关的变量
num=config.msg_dim
index = np.argmax([np.std(best_message['e%d'%(i,)]) for i in range(num)])
print("The variable with the maximum standard deviation:", index)

#提取用于拟合消息传递公式的特征
data_msg=best_message[['e%d'%(index,), 'dx', 'dy', 't', 'u_prev1', 'u_prev2', 'ux1', 'ux2', 'uy1', 'uy2']]
data_msg = data_msg.drop_duplicates() #去重
print(data_msg.head())
print(f'消息特征：{data_msg.shape}')

#提取用于拟合消息聚合公式的特征
data_node=best_message[['e%d'%(index,),'t','x2','y2']].copy()
data_node.rename(columns={f'e{index}': 'e0', 'x2': 'x', 'y2': 'y'}, inplace=True)
data_node = data_node.round(6)
data_node = data_node.drop_duplicates()
data_node_grouped = data_node.groupby(['x', 'y', 't'])['e0'].apply(list).reset_index()
max_len = 4
data_node_grouped['e_values'] = data_node_grouped['e0'].apply(lambda lst: (lst + [0]*max_len)[:max_len])
e_cols = pd.DataFrame(data_node_grouped['e_values'].tolist(), columns=[f'e{i}' for i in range(1, max_len+1)])
data_node = pd.concat([data_node_grouped[['x', 'y', 't']], e_cols], axis=1)

save_path=config.source_path
with np.load(save_path, allow_pickle=True) as data:
    values = data['solution']    
    t = data['t_eval']
    metadata = data['parameters'].item()
    data_node = bulid_aggr_feat(data,data_node)
data_node['e']=data_node['e1']+data_node['e2']+data_node['e3']+data_node['e4']
print(data_node.head())
print(f'聚合特征：{data_node.shape}')


# 数据采样
sample_size = 20000 
if len(data_msg) > sample_size:
    msg_sampled = data_msg.sample(n=sample_size, random_state=42) # 设置 seed 以保证结果可复现
    print(f"Sampling {sample_size} message data points.")
else:
    msg_sampled = data_msg

if len(data_node) > sample_size:
    node_sampled = data_node.sample(n=sample_size, random_state=42) 
    print(f"Sampling {sample_size} aggregate data points.")
else:
    node_sampled = data_node


# 步骤 2: 准备数据（X: 自变量, y: 目标 e%d）
# msg_feature_names = ['dx', 'dy', 't', 'u_prev1', 'u_prev2', 'ux1', 'ux2', 'uy1', 'uy2']
msg_feature_names = ['t', 'u_prev1', 'u_prev2', 'ux1', 'ux2', 'uy1', 'uy2']
node_feature_names = ['x', 'y', 't', 'ux', 'uy','e','u_prev']
# node_feature_names = ['x', 'y', 't']


X_msg = msg_sampled[msg_feature_names].values  # 形状: (n_samples, 5)
y_msg = msg_sampled['e%d'%(index,)].values  # 形状: (n_samples,)
X_node = node_sampled[node_feature_names].values
y_node = node_sampled['u'].values


# 步骤 3: 初始化 PySR 拟合器
# 可自定义：niterations=更多迭代，binary_operators=自定义运算符
SR_msg = PySRRegressor(
    niterations=100,           # 迭代次数（平衡速度/精度）
    populations=50,
    population_size=50,
    parsimony=0.01,
    binary_operators=["+", "*", "-", "/"],  # 支持加减乘除
    unary_operators=["sin","cos"], # 可选一元运算
    model_selection="best",    # 选最佳模型
    output_directory='D:/MyProject/GNN+Nesy/SGN/result/msg_formula',
    random_state=42,         
)
SR_node = PySRRegressor(
    niterations=100,           # 迭代次数（平衡速度/精度）
    populations=50,
    population_size=50,
    parsimony=0.01,
    binary_operators=["+", "*", "-", "/"],  # 支持加减乘除
    unary_operators=["sin","cos"], # 可选一元运算
    model_selection="best",    # 选最佳模型
    output_directory='D:/MyProject/GNN+Nesy/SGN/result/node_formula',
    random_state=42, 
)

# 步骤 4: 拟合并输出符号表达式
SR_msg.fit(X_msg, y_msg,variable_names=msg_feature_names)
SR_node.fit(X_node, y_node,variable_names=node_feature_names)

# 输出最佳表达式
formula_msg = SR_msg.get_best()
loss_msg= formula_msg['loss']
print(f"最佳消息传递表达式:{SR_msg.sympy()},损失为:{loss_msg}")
formula_node = SR_node.get_best()
loss_node= formula_node['loss']
print(f"最佳消息聚合表达式:{SR_node.sympy()},损失为:{loss_node}")
