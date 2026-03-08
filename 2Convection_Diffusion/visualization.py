############################################################
#                  Convection-Diffusion                    #
#                Visualization of Results                  #
############################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import config
from tool import create_interactive_comparison_viz
from tool import create_interactive_wave_viz

# Display GNN fitting results
# print(f"Model parameter settings:\nAggregation method:{config.aggr}\nHidden layer dimension:{config.hidden}\nMessage dimension:{config.msg_dim}")
# create_interactive_comparison_viz(train_path=config.source_path,
#                                     pred_path=f"result/pgn_prediction_2d{config.name}.npz",interval=0.05)

# # Display solver results
create_interactive_comparison_viz(train_path="data/convection_diffusion_2d(32).npz",
                                  pred_path=f"result/pgn_prediction_2d{config.name}_final.npz",
                                  interval=0.05)

# Display dataset
# save_path=config.source_path
# save_path=f"result/pgn_prediction_2d{config.name}.npz"
# with np.load(save_path, allow_pickle=True) as data:
#     values = data['solution']    
#     t = data['t_eval']
#     metadata = data['parameters'].item() 
#     print("Data shape:", values.shape)
#     print("Parameter information:", metadata)
#     # Extract Lx and Ly from metadata
#     Lx_val = metadata.get('Lx', 5.0) # Use .get() for safety
#     Ly_val = metadata.get('Ly', 5.0)
#     create_interactive_wave_viz(values, t, Lx=Lx_val, Ly=Ly_val, interval=0.05,vmax=1,vmin=0)