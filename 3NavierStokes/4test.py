############################################################
#                  Navier-Stokes Equation                  #
#       Step 4: Construct graph solver and test            #
############################################################


import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import  MessagePassing
import model
import config
import tool
import matplotlib.pyplot as plt


class CustomMsgFn(nn.Module):
    def __init__(self):
        super(CustomMsgFn, self).__init__()

    def forward(self, tmp):
        # x_i: [batch_edges, 3], x_j: [batch_edges, 3]
        # Concatenate inputs: cat([x_i, x_j], dim=-1) -> [batch_edges, 18]
        input_cat = tmp
        # print(input_cat.shape)
        
        # Unpack variables (according to formulas)
        # msg_feature_names = [x1 y1 t1 ux1 uy1 uxx1 uyy1 x_prev1 y_prev1],[x2 y2 t2 ux2 uy2 uxx2 uyy2 x_prev2 y_prev2]
        x1, y1, t, ux1, uy1, uxx1, uyy1, x_prev1, y_prev1 = input_cat[:, 0], input_cat[:, 1], input_cat[:, 2], \
                                     input_cat[:, 3], input_cat[:, 4], input_cat[:, 5], input_cat[:, 6], input_cat[:, 7], input_cat[:, 8]
        x2, y2, t2, ux2, uy2, uxx2, uyy2, x_prev2, y_prev2 = input_cat[:, 9], input_cat[:, 10], input_cat[:, 11], \
                                     input_cat[:, 12], input_cat[:, 13], input_cat[:, 14], input_cat[:, 15], input_cat[:, 16], input_cat[:, 17]            
        
        # Calculate msg1 
        msg1 = (-x_prev1 + x_prev2) * (-0.07927439)
        # Calculate msg2
        msg2 = (y_prev2 / (t / -0.006975147)) + -0.0055887355
        
        # Output [batch_edges, 2]
        msg = torch.stack([msg1, msg2], dim=-1)
        return msg

class CustomNodeFn(nn.Module):
    def __init__(self, output_dim=1):
        super(CustomNodeFn, self).__init__()
        self.output_dim = output_dim

    def forward(self, aggr_out):
        # aggr_out: [n, node_features + msg_dim], aggregated features and messages
        # node_feature_names = ['x', 'y', 't', 'ux', 'uy', 'uxx', 'uyy', 'x_prev', 'y_prev', 'e0', 'e1']

        # Unpack variables
        x, y, t, ux, uy, uxx, uyy, x_prev, y_prev, e0, e1 = aggr_out[:, 0], aggr_out[:, 1], aggr_out[:, 2], \
                                                           aggr_out[:, 3], aggr_out[:, 4], aggr_out[:, 5], \
                                                           aggr_out[:, 6], aggr_out[:, 7], aggr_out[:, 8], \
                                                           aggr_out[:, 9], aggr_out[:, 10]
        
        # Specific symbolic formulas
        out1 = (x_prev * 0.9897041) + (e0 * -0.011039681)
        out2 = ((e1 * -0.005140257) + y_prev) * 0.9898316
        
        out = torch.stack([out1, out2], dim=-1)
        return out


class InterpretableGN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, out_dim, edge_index, aggr='add'):
        super(InterpretableGN, self).__init__(aggr=aggr)  
        # Formula-defined message function
        self.msg_fnc = CustomMsgFn()
        # Formula-defined node update function
        self.node_fnc = CustomNodeFn(out_dim)
        self.edge_index = edge_index
        self.ndim = ndim
    
    def forward(self, x, edge_index):
        # x is [n, n_f], node feature tensor (n nodes, n_f features per node)
        # edge_index: edge index tensor, shape (2, num_edges)
        
        # propagate calls message(), aggregates based on aggr, and finally calls update()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
      
    def message(self, x_i, x_j):
        # x_i/x_j shape: [n_e, n_f]
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp shape: [E, 2 * in_channels]
        tmp = tmp.float()
        return self.msg_fnc(tmp)
    
    def update(self, aggr_out, x=None):
        # aggr_out shape: [n, msg_dim]
        tmp = torch.cat([x, aggr_out], dim=1)
        tmp = tmp.float()
        return self.node_fnc(tmp)

# Load dataset
# save_path=config.source_path
save_path="data/naviers_stokes_2d(32).npz"
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
X_feat = torch.from_numpy(X_feat)
Y_feat = torch.from_numpy(Y_feat)
edge_index = model.get_edge_index(Ny=N_y, Nx=N_x)

aggr = config.aggr
hidden = config.hidden
msg_dim = config.msg_dim  # Message dimension
dim = config.dim
out_dim = config.out_dim
n_f = len(X_feat[0][0]) # Feature dimension
print(f"nf={n_f}")

pgn = InterpretableGN(n_f, msg_dim, dim, out_dim, edge_index=edge_index, aggr=aggr).cuda()

# Set to evaluation mode
pgn.eval()

res_t = []
prev_pred = None  # Used to store predicted values from the previous step
for i in range(len(X_feat)):
    if i == 0:
        # First time step: Use the true initial state
        _input_feat = X_feat[i].clone()  # (Nx*Ny, 7)
    else:
        # Subsequent time steps: Replace the state features with the previous prediction
        xy_t_current = X_feat[i][:, :7].clone()  # First 7 dims: x, y, t, ux, uy, uxx, uyy

        _input_feat = torch.cat([xy_t_current, prev_pred], dim=1)  # Concatenate previous prediction (Nx*Ny, 2)
    
    _q = Data(
        x=_input_feat.cuda(),
        edge_index=edge_index.cuda()
    )
    res = pgn(_q.x, _q.edge_index)  # Predict current velocity field (Nx*Ny, 2)
    res_t.append(res.cpu())  # Move to CPU to prevent GPU memory buildup
    prev_pred = res.cpu()  # Update previous prediction for next iteration

# Define save path
prediction_save_path = f"result/pgn_prediction_2d{config.name}_final.npz"

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

# Calculate error: Align true and predicted solution shapes
u_true = values[1:]  # From t=1, corresponding to predictions (Nt, 2, Ny, Nx)
assert u_true.shape == u_solution_pred.shape, f"Shape mismatch: {u_true.shape} vs {u_solution_pred.shape}"

# ===== Error Calculation =====
mse = np.mean((u_solution_pred - u_true) ** 2)

# Error trend by time step for visualization
mse_t = np.mean((u_solution_pred - u_true) ** 2, axis=(1, 2, 3))  # (Nt,)

print(f"Global MSE: {mse:.6e}")
# print(f"Average step MSE: {np.mean(mse_t):.6e} ± {np.std(mse_t):.6e}")

plt.figure(figsize=(6,4))
plt.plot(t[1:], mse_t, label='MSE over time')
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()
plt.title('Prediction Error Evolution')
plt.grid(True)
plt.tight_layout()
plt.show()