############################################################
#                  Convection-Diffusion                    #
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
    需要从result中构造出[u,x,y,t,ux, uy,uxx,uyy u(x, y, t-1)]
    然后将x,y,t相等的项合并为featfeat[u,x,y,t,e1,e2,e3,e4,ux, uy,uxx, uyy, u(x, y, t-1)]
    注意：以data_node中x,y,t为主，即len(feat)=len(data_node)
    feat: (len(data_node), 13) -> [u(x, y, t),x, y, t, ux, uy,uxx,uyy, u(x, y, t-1),e1,e2,e3,e4]  # 排除初始步，只取从 i=1 开始
    """
    
    u = result['solution'] 
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
        # 计算二阶偏导 uxx (∂²u/∂x²) 和 uyy (∂²u/∂y²)
        uxx_2d = savgol_filter(u_prev_2d, 7, 2,deriv=2,delta=dx,axis=1)  # 平滑处理
        uyy_2d = savgol_filter(u_prev_2d, 7, 2,deriv=2,delta=dy,axis=0)  # 平滑处理

        u_prev_2d_smooth = savgol_filter(u_prev_2d, 7, 2, deriv=0, axis=0)
        prev_u = u_prev_2d_smooth.flatten() # (Nx*Ny,) - 使用平滑后的 u
        feat_step = pd.DataFrame({
            'x': X.flatten(),
            'y': Y.flatten(),
            't': np.full(X.size, np.round(t, 6)),  # 时间取当前步
            'u': u[i].flatten(),  # 当前 x_solution
            'ux': ux_2d.flatten(),
            'uy': uy_2d.flatten(),
            'uxx': uxx_2d.flatten(),
            'uyy': uyy_2d.flatten(),
            'u_prev': prev_u,  # 前一步 x_solution
        })

        feat_list.append(feat_step)
    # 拼接所有时间步
    feat_df = pd.concat(feat_list, ignore_index=True)
    # 对浮点数进行标准化取整（防止精度误差）
    feat_df = feat_df.round(6)
    data_node = data_node.round(6)

    # 合并 PDE 特征和 message 特征
    merged = pd.merge(
        data_node,
        feat_df,
        on=['x', 'y', 't'],
        how='left'
    )

    # 重新排序列
    merged = merged[['u', 'x', 'y', 't', 'ux', 'uy', 'uxx', 'uyy', 'u_prev', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9']]
    return merged

# 步骤 1: 加载保存的数据
best_message =pkl.load(open(f'result/messages_best{config.name}.pkl', 'rb'))
print("Loaded DataFrame shape:", best_message.shape)
# print(best_message.head()) #查看完整的特征

#提取用于拟合消息传递公式的特征
# 分量1
data_msg_1 = best_message[['e0', 'dx', 'dy', 't', 'u_prev1', 'u_prev2', 'ux1', 'ux2', 'uy1', 'uy2', 'uxx1', 'uxx2', 'uyy1', 'uyy2']]
print(data_msg_1.head())
print(f'消息特征：{data_msg_1.shape}')
# 分量2
data_msg_2=best_message[['e1', 'dx', 'dy', 't', 'u_prev1', 'u_prev2', 'ux1', 'ux2', 'uy1', 'uy2', 'uxx1', 'uxx2', 'uyy1', 'uyy2']]
data_msg_2 = data_msg_2.drop_duplicates() #去重
print(data_msg_2.head())
print(f'消息特征：{data_msg_2.shape}')

save_path=config.source_path
# save_path=f"result/pgn_prediction_2d{config.name}.npz"
with np.load(save_path, allow_pickle=True) as data:  
    values = data['solution']
    t = data['t_eval']
    metadata = data['parameters'].item()

#提取用于拟合消息聚合公式的特征
data_node = best_message[['e0', 'e1','t','x2','y2']].copy()
data_node.rename(columns={f'x2': 'x', 'y2': 'y'}, inplace=True)
data_node = data_node.round(6)
data_node = data_node.drop_duplicates()
# 按 (x, y, t) 分组，收集 e0 和 e1 的列表
data_node_grouped1 = data_node.groupby(['x', 'y', 't'])['e0'].apply(list).reset_index()
data_node_grouped2 = data_node.groupby(['x', 'y', 't'])['e1'].apply(list).reset_index()

max_len = 4
# 填充列表到 max_len，用 0 补齐并截断
data_node_grouped1['e_values'] = data_node_grouped1['e0'].apply(lambda lst: (lst + [0]*max_len)[:max_len])
data_node_grouped2['e_values'] = data_node_grouped2['e1'].apply(lambda lst: (lst + [0]*max_len)[:max_len])
# 展开为列：e_cols1 (from e0: e1-e4)，e_cols2 (from e1: e5-e8，避免冲突)
e_cols1 = pd.DataFrame(data_node_grouped1['e_values'].tolist(), columns=[f'e{i+1}' for i in range(1, max_len+1)])
e_cols2 = pd.DataFrame(data_node_grouped2['e_values'].tolist(), columns=[f'e{i+4+1}' for i in range(1, max_len+1)])
#  共8列: x, y, t, (e2, e3, e4, e5), (e6, e7, e8, e9)
data_node = pd.concat([data_node_grouped1[['x', 'y', 't']], e_cols1, e_cols2], axis=1)


with np.load(save_path, allow_pickle=True) as data:  
    t = data['t_eval']
    metadata = data['parameters'].item()
    data_node = bulid_aggr_feat(data,data_node)
data_node['e0']=data_node['e2']+data_node['e3']+data_node['e4']+data_node['e5']
data_node['e1']=data_node['e6']+data_node['e7']+data_node['e8']+data_node['e9']
data_node_x=data_node[['u', 'x', 'y', 't', 'ux', 'uy', 'uxx', 'uyy', 'u_prev', 'e0', 'e1']]

print(data_node_x.head())
print(f'聚合特征：{data_node_x.shape}')

# 数据采样
sample_size = 20000 
if len(data_msg_1) > sample_size:
    msg_sampled_1 = data_msg_1.sample(n=sample_size, random_state=42) # 设置 seed 以保证结果可复现
    print(f"Sampling {sample_size} message data points.")
else:
    msg_sampled_1 = data_msg_1

if len(data_msg_2) > sample_size:
    msg_sampled_2 = data_msg_2.sample(n=sample_size, random_state=42) 
    print(f"Sampling {sample_size} message data points.")
else:
    msg_sampled_2 = data_msg_2

if len(data_node_x) > sample_size:
    node_sampled_x = data_node_x.sample(n=sample_size, random_state=42) 
    print(f"Sampling {sample_size} aggregate data points.")
else:
    node_sampled_x = data_node_x

# 步骤 2: 准备数据（X: 自变量, y: 目标 e%d）
# msg1待选：  e0,  dx, dy, t, u_prev1,  u_prev1,  ux1,  ux2,  uy1,  uy2, uxx1, uxx2, uyy1, uyy2
msg1_feature_names = ['t', 'u_prev1', 'u_prev2', 'ux1',  'ux2',  'uy1',  'uy2', 'uxx1', 'uxx2', 'uyy1', 'uyy2']
# msg2待选：  e1,  dx, dy, t, x_prev1,  y_prev1,  ux1,  ux2,  uy1,  uy2, uxx1, uxx2, uyy1, uyy2
msg2_feature_names = ['t', 'u_prev1', 'u_prev2', 'ux1',  'ux2',  'uy1',  'uy2', 'uxx1', 'uxx2', 'uyy1', 'uyy2']
# nodeX待选： u, x,  y,  t,  ux, uy,uxx, uyy, u_prev, e0, e1
node_x_feature_names = ['x', 'y', 't', 'ux','uy', 'u_prev', 'e0', 'e1']

X_msg1 = msg_sampled_1[msg1_feature_names].values  # 形状: (n_samples, 5)
X_msg2 = msg_sampled_2[msg2_feature_names].values
y_msg1 = msg_sampled_1['e0'].values  # 形状: (n_samples,)
y_msg2 = msg_sampled_2['e1'].values

X_node_input = node_sampled_x[node_x_feature_names].values
X_node_output = node_sampled_x['u'].values


# 步骤 3: 初始化 PySR 拟合器
# 可自定义：niterations=更多迭代，binary_operators=自定义运算符
SR_msg1 = PySRRegressor(
    niterations=100,           # 迭代次数（平衡速度/精度）
    populations=50,
    population_size=50,
    parsimony=0.01,
    binary_operators=["+", "*", "-", "/"],  # 支持加减乘除
    unary_operators=["sin","cos"], # 可选一元运算
    model_selection="best",    # 选最佳模型
    output_directory='D:/MyProject/GNN+Nesy/SGN/result/msg_formula/msg1',
    random_state=42,
)
SR_msg2 = PySRRegressor(
    niterations=100,           # 迭代次数（平衡速度/精度）
    populations=50,
    population_size=50,
    parsimony=0.01,
    binary_operators=["+", "*", "-", "/"],  # 支持加减乘除
    unary_operators=["sin","cos"], # 可选一元运算
    model_selection="best",    # 选最佳模型
    output_directory='D:/MyProject/GNN+Nesy/SGN/result/msg_formula/msg2',
    random_state=42
)
def square(x):
    return x * x

SR_node_x = PySRRegressor(
    niterations=100,           # 迭代次数（平衡速度/精度）
    populations=50,
    population_size=50,
    parsimony=0.01,
    binary_operators=["+", "*", "-", "/",],  # 支持加减乘除
    unary_operators=["sin","cos","exp","square"], # 可选一元运算
    model_selection="best",    # 选最佳模型
    output_directory='D:/MyProject/GNN+Nesy/SGN/result/node_formula',
    random_state=42,
)


# 步骤 4: 拟合并输出符号表达式
# SR_msg1.fit(X_msg1, y_msg1,variable_names=msg1_feature_names)
# SR_msg2.fit(X_msg2, y_msg2,variable_names=msg2_feature_names)
SR_node_x.fit(X_node_input, X_node_output,variable_names=node_x_feature_names)


# 输出最佳表达式
# formula_msg1 = SR_msg1.get_best()
# loss_msg1= formula_msg1['loss']
# print(f"最佳消息传递表达式1:{SR_msg1.sympy()},损失为:{loss_msg1}")
# formula_msg2 = SR_msg2.get_best()
# loss_msg2= formula_msg2['loss']
# print(f"最佳消息传递表达式2:{SR_msg2.sympy()},损失为:{loss_msg2}")

formula_node_x = SR_node_x.get_best()
loss_node_x= formula_node_x['loss']
print(f"最佳消息聚合表达式:{SR_node_x.sympy()},损失为:{loss_node_x}")
