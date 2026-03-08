############################################################
#                       Wave Equation                      #
#           Step 3: Extract formulas using PySR            #
############################################################

from pysr import PySRRegressor
import pickle as pkl
import numpy as np
import pandas as pd
import config
from scipy.signal import savgol_filter

def bulid_aggr_feat(result,data_node):
    """
    data_node features are [x, y, t, e1, e2, e3, e4]
    I need to construct [u, x, y, t, ux, uy, u(x, y, t-1)] from result
    Then merge items with equal x, y, t into feat[u, x, y, t, e1, e2, e3, e4, ux, uy, u(x, y, t-1)]
    Note: prioritize x, y, t in data_node, i.e., len(feat)=len(data_node)
    feat: (len(data_node), 7+4) -> [u(x, y, t), x, y, t, ux, uy, u(x, y, t-1), e1, e2, e3, e4]  # Exclude initial step, start from i=1
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

    # Generate spatial coordinate grid
    X, Y = np.meshgrid(x_arr, y_arr, indexing='xy')  # (Ny, Nx)

    # Initialize containers
    feat_list = []
    
    # Iterate through time steps, starting from i=1
    for i in range(1, original_Nt_plus1):
        t = t_arr[i]  
        # Spatial features for each time step [x, y, t]
        # xy_t = np.stack([X.flatten(), Y.flatten(), np.full(X.size, t)], axis=1)  # (Nx*Ny, 3)
        
        # u value from the previous time step: u[i-1], keep as 2D for gradient calculation
        u_prev_2d = u[i-1]  # (Ny, Nx)
        # Calculate 1st order partial derivative ux (∂u/∂x) and uy (∂u/∂y)
        # ux_2d = np.gradient(u_prev_2d, x_arr, axis=1)  # (Ny, Nx), along x direction
        # uy_2d = np.gradient(u_prev_2d, y_arr, axis=0)  # (Ny, Nx), along y direction
        ux_2d = savgol_filter(u_prev_2d, 7, 2, deriv=1, delta=dx, axis=1)  # Smoothing processing
        uy_2d = savgol_filter(u_prev_2d, 7, 2, deriv=1, delta=dy, axis=0)  # Smoothing processing
        u_temp = savgol_filter(u_prev_2d, 7, 2, deriv=0, axis=1)
        u_prev_2d_smooth = savgol_filter(u_temp, 7, 2, deriv=0, axis=0)
        prev_u = u_prev_2d_smooth.flatten() # (Nx*Ny,) - Use smoothed u
        feat_step = pd.DataFrame({
            'x': X.flatten(),
            'y': Y.flatten(),
            't': np.full(X.size, np.round(t, 6)),  # Use current step for time
            'u': u[i].flatten(),
            'ux': ux_2d.flatten(),
            'uy': uy_2d.flatten(),
            # 'u_prev': u_prev_2d.flatten()
            'u_prev': prev_u
        })
        feat_list.append(feat_step)
    # Concatenate all time steps
    feat_df = pd.concat(feat_list, ignore_index=True)
    # Standardize and round floating-point numbers (prevent precision errors)
    feat_df = feat_df.round(6)
    data_node = data_node.round(6)

    print(feat_df.head())
    print("feat_df shape:", feat_df.shape)
    print(data_node.head())
    print("data_node shape:", data_node.shape)

    # Merge PDE features and message features
    merged = pd.merge(
        data_node,
        feat_df,
        on=['x', 'y', 't'],
        how='left'
    )

    # Reorder columns
    merged = merged[['u', 'x', 'y', 't', 'ux', 'uy', 'u_prev', 'e1', 'e2', 'e3', 'e4']]
    return merged

# Step 1: Load saved data
best_message = pkl.load(open(f'result/messages_best{config.name}.pkl', 'rb'))
print("Loaded DataFrame shape:", best_message.shape)
print(best_message.head()) # View complete features

# Get related variables through maximum standard deviation
num=config.msg_dim
index = np.argmax([np.std(best_message['e%d'%(i,)]) for i in range(num)])
print("The variable with the maximum standard deviation:", index)

# Extract features used to fit the message passing formula
data_msg=best_message[['e%d'%(index,), 'dx', 'dy', 't', 'u_prev1', 'u_prev2', 'ux1', 'ux2', 'uy1', 'uy2']]
data_msg = data_msg.drop_duplicates() # Deduplicate
print(data_msg.head())
print(f'Message features: {data_msg.shape}')

# Extract features used to fit the message aggregation formula
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
print(f'Aggregation features: {data_node.shape}')



# Data sampling
sample_size = 20000 
if len(data_msg) > sample_size:
    msg_sampled = data_msg.sample(n=sample_size, random_state=42) # Set seed to ensure reproducibility
    print(f"Sampling {sample_size} message data points.")
else:
    msg_sampled = data_msg

if len(data_node) > sample_size:
    node_sampled = data_node.sample(n=sample_size, random_state=42) 
    print(f"Sampling {sample_size} aggregate data points.")
else:
    node_sampled = data_node


# Step 2: Prepare data (X: independent variables, y: target e%d)
# msg_feature_names = ['dx', 'dy', 't', 'u_prev1', 'u_prev2', 'ux1', 'ux2', 'uy1', 'uy2']
msg_feature_names = ['t', 'u_prev1', 'u_prev2', 'ux1', 'ux2', 'uy1', 'uy2']
node_feature_names = ['x', 'y', 't', 'ux', 'uy','e','u_prev']
# node_feature_names = ['x', 'y', 't']


X_msg = msg_sampled[msg_feature_names].values  # Shape: (n_samples, 5)
y_msg = msg_sampled['e%d'%(index,)].values  # Shape: (n_samples,)
X_node = node_sampled[node_feature_names].values
y_node = node_sampled['u'].values


# Step 3: Initialize PySR regressor
# Customizable: niterations=more iterations, binary_operators=custom operators
SR_msg = PySRRegressor(
    niterations=100,           # Number of iterations (balance speed/accuracy)
    populations=50,
    population_size=50,
    parsimony=0.01,
    binary_operators=["+", "*", "-", "/"],  # Support addition, subtraction, multiplication, division
    unary_operators=["sin","cos"], # Optional unary operators
    model_selection="best",    # Select the best model
    output_directory='D:/MyProject/GNN+Nesy/SGN/result/msg_formula',
    random_state=42,         
)
SR_node = PySRRegressor(
    niterations=100,           # Number of iterations (balance speed/accuracy)
    populations=50,
    population_size=50,
    parsimony=0.01,
    binary_operators=["+", "*", "-", "/"],  # Support addition, subtraction, multiplication, division
    unary_operators=["sin","cos"], # Optional unary operators
    model_selection="best",    # Select the best model
    output_directory='D:/MyProject/GNN+Nesy/SGN/result/node_formula',
    random_state=42, 
)

# Step 4: Fit and output symbolic expression
SR_msg.fit(X_msg, y_msg,variable_names=msg_feature_names)
SR_node.fit(X_node, y_node,variable_names=node_feature_names)

# Output best expression
formula_msg = SR_msg.get_best()
loss_msg= formula_msg['loss']
print(f"Best message passing expression: {SR_msg.sympy()}, Loss: {loss_msg}")
formula_node = SR_node.get_best()
loss_node= formula_node['loss']
print(f"Best message aggregation expression: {SR_node.sympy()}, Loss: {loss_node}")