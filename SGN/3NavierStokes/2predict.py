############################################################
#                  Navier-Stokes Equation                  #
#                Step 2: Data Fitting with GNN             #
############################################################

import model
import numpy as np
from model import PGN
from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt
import config
import tool

# Load dataset
save_path=config.source_path
with np.load(save_path, allow_pickle=True) as data:
    values = data['solution']
    data_x = data['x']
    data_y = data['y']
    t = data['t_eval']
    metadata = data['parameters'].item()
    X_feat, Y_feat = tool.build_feat_2_SG(data)

# Grid granularity
N_x = metadata['Nx']
N_y = metadata['Ny']
n = N_x * N_y
X_feat=torch.from_numpy(X_feat)
Y_feat=torch.from_numpy(Y_feat)
edge_index = model.get_edge_index(Ny=N_y,Nx=N_x)
aggr = config.aggr
hidden = config.hidden
msg_dim = config.msg_dim  # Message dimension
dim=config.dim
out_dim=config.out_dim
n_f = len(X_feat[0][0]) # Feature dimension
print(f"nf={n_f}")

pgn = PGN(n_f, msg_dim, dim, out_dim, hidden=hidden, edge_index=edge_index, aggr=aggr).cuda()
pgn.load_state_dict(torch.load(f'result/models_best{config.name}.pth',map_location='cuda'))

# Set to evaluation mode
pgn.eval()

res_t=[]
prev_pred = None  # Used to store the predicted value from the previous step
for i in range(len(X_feat)):
    if i == 0:
        # First time step: Use the true initial u_prev
        _input_feat = X_feat[i].clone()  # (Nx*Ny, 7)
    else:
        # Subsequent time steps: Replace the corresponding dimension with the prediction from the previous step
        xy_t_current = X_feat[i][:, :7].clone()  # First 7 dimensions: x, y, t, ux, uy

        _input_feat = torch.cat([xy_t_current, prev_pred], dim=1)  # Concatenate prediction (Nx*Ny, 5) x_prev y_prev 
    
    _q = Data(
        x=_input_feat.cuda(),
        edge_index=edge_index.cuda()
    )
    res = pgn(_q.x, _q.edge_index)  # Predict current u (Nx*Ny,)
    res_t.append(res.cpu())  # Move to CPU for storage to prevent GPU memory accumulation
    prev_pred = res.cpu()  # Update the prediction for the previous step (for the next round)

# Define save path
prediction_save_path = f"result/pgn_prediction_2d{config.name}.npz"

u_pred_tensors = torch.stack(res_t).cpu()
u_pred_flat = u_pred_tensors.detach().numpy()
Nt = len(t) - 1
u_pred = np.zeros((Nt, N_y, N_x))
v_pred = np.zeros((Nt, N_y, N_x))
for i in range(Nt):
    u_pred[i] = u_pred_flat[i, :, 0].reshape(N_y, N_x)
    v_pred[i] = u_pred_flat[i, :, 1].reshape(N_y, N_x)
u_solution_pred = np.stack([u_pred, v_pred], axis=1)  # (Nt, 2, Ny, Nx)

result = {
    'solution': u_solution_pred,
    'x_solution': u_pred,
    'y_solution': v_pred,
    'x': data_x,
    'y': data_y,
    't_eval': t-1, 
    'parameters': metadata 
}

np.savez(prediction_save_path, **result)
print(f"Results saved to {prediction_save_path}")


# Calculate error: Align shapes of the true solution and predicted solution
u_true = values[1:]  # Start from t=1, corresponding to the prediction (Nt, Ny, Nx)
assert u_true.shape == u_solution_pred.shape, f"Shape mismatch: {u_true.shape} vs {u_solution_pred.shape}"

# ===== Error Calculation Section =====
mse = np.mean((u_solution_pred - u_true) ** 2)

# Calculate error trend per time step (for visual analysis)
mse_t = np.mean((u_solution_pred - u_true) ** 2, axis=(1, 2, 3))  # (Nt,)

print(f"Global MSE: {mse:.6e}")
# print(f"Average MSE per step: {np.mean(mse_t):.6e} ± {np.std(mse_t):.6e}")


plt.figure(figsize=(6,4))
plt.plot(t[1:], mse_t, label='MSE over time')
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()
plt.title('Prediction Error Evolution')
plt.grid(True)
plt.tight_layout()
plt.show()