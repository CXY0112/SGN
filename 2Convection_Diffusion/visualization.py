############################################################
#                  Convection-Diffusion                    #
#                       可视化结果                          #
############################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path
import sys

# 将项目根目录（上一层）加入导入路径
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from tool import create_interactive_comparison_viz
from tool import create_interactive_wave_viz

# GNN拟合结果展示
# print(f"模型参数设置：\n聚合方式:{config.aggr}\n隐藏层维度:{config.hidden}\n消息维度:{config.msg_dim}")
# create_interactive_comparison_viz(train_path=config.source_path,
#                                     pred_path=f"result/pgn_prediction_2d{config.name}.npz",interval=0.05)

# # 求解器结果展示
create_interactive_comparison_viz(train_path="data/convection_diffusion_2d(32).npz",
                                  pred_path=f"result/convection_diffusion\convection_diffusion（0005）/pgn_prediction_2d(con_diff_add_0002_dnoise)_final.npz",
                                  interval=0.05)

# 数据集展示
# save_path=config.source_path
# save_path=f"result/pgn_prediction_2d{config.name}.npz"
# with np.load(save_path, allow_pickle=True) as data:
#     values = data['solution']    
#     t = data['t_eval']
#     metadata = data['parameters'].item() 
#     print("数据形状:", values.shape)
#     print("参数信息:", metadata)
#     # 从 metadata 中提取 Lx 和 Ly
#     Lx_val = metadata.get('Lx', 5.0) # 使用 .get() 确保安全
#     Ly_val = metadata.get('Ly', 5.0)
#     create_interactive_wave_viz(values, t, Lx=Lx_val, Ly=Ly_val, interval=0.05,vmax=1,vmin=0)
