#######################################
#           Model Configuration       #
#######################################

# 2D Wave Equation Parameters
source_path="data/wave_solution_2d(32)_noise_0.1.npz"
aggr = 'add' # options: sum, mean, min, max, mul, etc.
hidden = 30  # hidden layer dimension
msg_dim = 1  # message dimension
dim=2        # 2D or 3D problem
out_dim=1
columns='x%d y%d t%d ux%d uy%d u_prev%d'                        # input features
loss_type = '_l1_'
name=f"(wave_{aggr}_test)"               # for file naming
# name=f"(wave_{aggr}_01_dnoise)"  

# # 2D Convection-Diffusion Equation Parameters
# source_path="data/convection_diffusion_2d(32)_noise_0.05.npz"
# # source_path="data/convection_diffusion_2d(32).npz"
# aggr = 'add' # options: sum, mean, min, max, mul, etc.
# hidden = 30  # hidden layer dimension
# msg_dim = 2  # message dimension
# dim=2        # 2D or 3D problem
# out_dim=1
# columns='x%d y%d t%d ux%d uy%d uxx%d uyy%d u_prev%d'                        # input features
# loss_type = '_l1_'
# name=f"(con_diff_{aggr}_test)"
# # name=f"(con_diff_{aggr}_005_dnoise)"  

# # Navier-Stokes Equation
# source_path="data/naviers_stokes_2d(32)_noise_0.01.npz"
# aggr = 'add' # options: sum, mean, min, max, mul, etc.
# hidden = 30  # hidden layer dimension
# msg_dim = 2  # message dimension
# dim=2        # 2D or 3D problem
# out_dim=2    # output dimension is 2
# loss_type = '_l1_'
# columns='x%d y%d t%d ux%d uy%d uxx%d uyy%d x_prev%d y_prev%d'               # input features
# # name=f"(ns_{aggr}_test)" 
# name=f"(ns_{aggr}_001_dnoise)"  

