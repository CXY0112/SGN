############################################################
#                  Convection-Diffusion                    #
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
        # Concatenate inputs: cat([x_i, x_j], dim=-1) -> [batch_edges, 6]
        input_cat = tmp

        # Unpack variables (according to formula)
        # msg_feature_names = [x1, y1, t1, ux1, uy1, u1(x, y, t-1)],[x2, y2, t2, ux2, uy2, u2(x, y, t-1)]
        x1, y1, t, ux1, uy1, uxx1, uyy1, u_prev1, x2, y2, t2, ux2, uy2, uxx2, uyy2, u_prev2 = input_cat[:, 0], input_cat[:, 1], input_cat[:, 2], \
                                                                 input_cat[:, 3], input_cat[:, 4], input_cat[:, 5], input_cat[:, 6], input_cat[:, 7], \
                                                                 input_cat[:, 8], input_cat[:, 9], input_cat[:, 10], input_cat[:, 11], \
                                                                 input_cat[:, 12], input_cat[:, 13], input_cat[:, 14], input_cat[:, 15]
        # dx=x1-x2
        # dy=y1-y2

        # Calculate msg1 = sin(x1 + x2) + x3 * y3
        msg1 = u_prev2 * (torch.cos(np.sin(-0.037906747) * ((ux1 / -0.33668214) + uyy1)) * 0.39657518)
        # Calculate msg2 
        msg2 = uy2*(-0.007133365) - (-0.07567513)*(u_prev2 - 0.0005025013*uyy1*uyy1)
        
        # Output [batch_edges, 2]
        msg = torch.stack([msg1,msg2], dim=-1)
        return msg

class CustomNodeFn(nn.Module):
    def __init__(self, output_dim=1):
        super(CustomNodeFn, self).__init__()
        self.output_dim=output_dim

    def forward(self, aggr_out):
        # aggr_out: [n, msg_dim=2], aggregated message
        # node_feature_names = [x, y, t, ux, uy, u(x, y, t-1)],['e']
        # Unpack variables
        x, y, t, ux, uy, uxx, uyy, u_prev, e0, e1 = aggr_out[:, 0], aggr_out[:, 1], aggr_out[:, 2], aggr_out[:, 3], aggr_out[:, 4], aggr_out[:, 5], aggr_out[:, 6], aggr_out[:, 7], aggr_out[:, 8], aggr_out[:, 9]    
        
        # Specific formula
        # out = u_prev * (0.98885995 ** ((u_prev ** ((-1.0377206 + u_prev) * e1)) * (uy + ((0.893268 ** ((torch.exp(ux) - e0) * (1.1708163 + (0.40441316 + 1.1708163)))) * (-0.2606126 + ux)))))
        out = u_prev + ((ux + ((torch.sin(u_prev / 0.021259548) / (torch.square(ux) + 0.21166894)) + -0.30247244)) * -0.0043281894)
        if self.output_dim == 1:
            out = out.unsqueeze(-1)
        return out


class InterpretableGN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim,out_dim, edge_index, aggr='add'):
        super(InterpretableGN, self).__init__(aggr=aggr)  
        # Explicit formula-defined message generation function
        self.msg_fnc = CustomMsgFn()
        # Explicit formula-defined message aggregation function
        self.node_fnc = CustomNodeFn(out_dim)
        self.edge_index = edge_index
        self.ndim = ndim
    
    def forward(self, x, edge_index):
        # x is [n, n_f], node feature tensor, where n is the number of nodes, n_f is the feature dimension per node
        # edge_index: edge index tensor, shape (2, num_edges)
        x = x # ?

        # propagate calls message(), then aggregates based on aggr, and finally calls update()
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

# Load dataset
# save_path=config.source_path
save_path='data/convection_diffusion_2d(32).npz'
with np.load(save_path, allow_pickle=True) as data:
    values = data['solution']
    data_x = data['x']
    data_y = data['y']
    t = data['t_eval']
    metadata = data['parameters'].item()
    X_feat, Y_feat = tool.build_feat_SG_2(data,7,2)
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

pgn = InterpretableGN(n_f, msg_dim, dim, out_dim, edge_index=edge_index, aggr=aggr).cuda()

# Set to evaluation mode
pgn.eval()

res_t=[]
prev_pred = None  # Used to store the predicted value from the previous step
for i in range(len(X_feat)):
    if i == 0:
        # First time step: Use the true initial u_prev
        _input_feat = X_feat[i].clone()  # (Nx*Ny, 8)
    else:
        # Subsequent time steps: Replace the corresponding dimension with the prediction from the previous step
        xy_t_current = X_feat[i][:, :7].clone()  # First 7 dimensions: x, y, t, etc. (Nx*Ny, 7)

        _input_feat = torch.cat([xy_t_current, prev_pred], dim=1)  # Concatenate predicted u_prev (Nx*Ny, 8)
    
    _q = Data(
        x=_input_feat.cuda(),
        edge_index=edge_index.cuda()
    )
    res = pgn(_q.x, _q.edge_index)  # Predict current u (Nx*Ny,)
    res_t.append(res.cpu())  # Move to CPU for storage to prevent GPU memory accumulation
    prev_pred = res.cpu()  # Update the prediction for the previous step (for the next round)

# Define save path
prediction_save_path = f"result/pgn_prediction_2d{config.name}_final.npz"

u_pred_tensors = torch.stack(res_t).cpu()
u_pred_flat = u_pred_tensors.detach().numpy()
u_solution_pred = u_pred_flat.reshape(len(t)-1, N_y, N_x)

result = {
    'solution': u_solution_pred,
    'x': data_x,
    'y': data_y,
    't_eval': t[1:], 
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
mse_t = np.mean((u_solution_pred - u_true) ** 2, axis=(1, 2))  # (Nt,)

print(f"Global MSE: {mse:.6e}")
# print(f"Average MSE per step: {np.mean(mse_t):.6e} ± {np.std(mse_t):.6e}")

# ===== Save Prediction Results and Errors =====
result = {
    'solution': u_solution_pred,
    'x': data_x,
    'y': data_y,
    't_eval': t[1:],  # Note: Align with the predicted steps here
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