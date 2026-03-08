############################################################
#                  Navier-Stokes Equation                  #
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
    data_node features are [x,y,t,e1,e2,e3,e4]
    Need to construct [values_x,values_y,x,y,t,ux, uy,uxx,uyy, x_pred, y_pred] from result
    Then merge items with equal x,y,t into feat[values_x,values_y,x,y,t,e1,e2,e3,e4,ux, uy, x_pred, y_pred]
    Note: prioritize x,y,t in data_node, i.e. len(feat)=len(data_node)
    feat: (len(data_node), 7+6) -> [values_x,values_y,x, y, t, ux, uy, uxx, uyy, x_pred,y_pred,e1,e2,e3,e4]  # Exclude initial step, start from i=1
    """
    
    values_x=result['x_solution'] 
    values_y=result['y_solution']
    x_arr = result['x']       # (Nx,)
    y_arr = result['y']       # (Ny,)
    t_arr = result['t_eval']  # (Nt+1,)

    original_Nt_plus1, Ny, Nx = values_x.shape
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
        x_prev_2d = values_x[i-1]  # (Ny, Nx)
        y_prev_2d = values_y[i-1]  # (Ny, Nx)

        # Calculate 1st order partial derivative ux = ∂(x_solution(t-1))/∂x and uy = ∂(y_solution(t-1))/∂y
        ux_2d = savgol_filter(x_prev_2d, 7, 2,deriv=1,delta=dx,axis=1)  # Smoothing processing
        uy_2d = savgol_filter(y_prev_2d, 7, 2,deriv=1,delta=dy,axis=0)  # Smoothing processing

        # Calculate 2nd order partial derivative uxx = ∂²(x_solution(t-1))/∂x² and uyy = ∂²(y_solution(t-1))/∂y²
        uxx_2d = savgol_filter(x_prev_2d, 7, 2,deriv=2,delta=dx,axis=1)  # Smoothing processing
        uyy_2d = savgol_filter(y_prev_2d, 7, 2,deriv=2,delta=dy,axis=0)  # Smoothing processing

        feat_step = pd.DataFrame({
            'x': X.flatten(),
            'y': Y.flatten(),
            't': np.full(X.size, np.round(t, 6)),  # Use current step for time
            'values_x': values_x[i].flatten(),  # Current x_solution
            'values_y': values_y[i].flatten(),  # Current y_solution
            'ux': ux_2d.flatten(),
            'uy': uy_2d.flatten(),
            'uxx': uxx_2d.flatten(),
            'uyy': uyy_2d.flatten(),
            'x_prev': x_prev_2d.flatten(),  # Previous step x_solution
            'y_prev': y_prev_2d.flatten()   # Previous step y_solution
        })

        feat_list.append(feat_step)
    # Concatenate all time steps
    feat_df = pd.concat(feat_list, ignore_index=True)
    # Standardize and round floating-point numbers (prevent precision errors)
    feat_df = feat_df.round(6)
    data_node = data_node.round(6)

    # Merge PDE features and message features
    merged = pd.merge(
        data_node,
        feat_df,
        on=['x', 'y', 't'],
        how='left'
    )

    # Reorder columns
    merged = merged[['values_x', 'values_y', 'x', 'y', 't', 'ux', 'uy','uxx', 'uyy', 'x_prev', 'y_prev', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9']]
    return merged

# Step 1: Load saved data
best_message =pkl.load(open(f'result/messages_best{config.name}.pkl', 'rb'))
print("Loaded DataFrame shape:", best_message.shape)
# print(best_message.head()) # View complete features

# Extract features used to fit the message passing formula
# Component 1
data_msg_1 = best_message[['e0', 'dx', 'dy', 't', 'x_prev1', 'y_prev1', 'x_prev2', 'y_prev2', 'ux1', 'ux2', 'uy1', 'uy2', 'uxx1', 'uxx2', 'uyy1', 'uyy2']]
data_msg_1 = data_msg_1.drop_duplicates() # Deduplicate
print(data_msg_1.head())
print(f'Message features 1: {data_msg_1.shape}')
# Component 2
data_msg_2=best_message[['e1', 'dx', 'dy', 't', 'x_prev1', 'y_prev1', 'x_prev2', 'y_prev2', 'ux1', 'ux2', 'uy1', 'uy2', 'uxx1', 'uxx2', 'uyy1', 'uyy2']]
data_msg_2 = data_msg_2.drop_duplicates() # Deduplicate
print(data_msg_2.head())
print(f'Message features 2: {data_msg_2.shape}')

save_path="data/naviers_stokes_2d.npz"
with np.load(save_path, allow_pickle=True) as data:  
    value_x = data['x_solution']
    value_y = data['y_solution']
    t = data['t_eval']
    metadata = data['parameters'].item()

# Extract features used to fit the message aggregation formula
data_node = best_message[['e0', 'e1','t','x2','y2']].copy()
data_node.rename(columns={f'x2': 'x', 'y2': 'y'}, inplace=True)
data_node = data_node.round(6)
data_node = data_node.drop_duplicates()
# Group by (x, y, t), collect lists of e0 and e1
data_node_grouped1 = data_node.groupby(['x', 'y', 't'])['e0'].apply(list).reset_index()
data_node_grouped2 = data_node.groupby(['x', 'y', 't'])['e1'].apply(list).reset_index()

max_len = 4
# Pad lists to max_len, fill with 0 and truncate
data_node_grouped1['e_values'] = data_node_grouped1['e0'].apply(lambda lst: (lst + [0]*max_len)[:max_len])
data_node_grouped2['e_values'] = data_node_grouped2['e1'].apply(lambda lst: (lst + [0]*max_len)[:max_len])
# Expand into columns: e_cols1 (from e0: e1-e4), e_cols2 (from e1: e5-e8, avoid conflicts)
e_cols1 = pd.DataFrame(data_node_grouped1['e_values'].tolist(), columns=[f'e{i+1}' for i in range(1, max_len+1)])
e_cols2 = pd.DataFrame(data_node_grouped2['e_values'].tolist(), columns=[f'e{i+4+1}' for i in range(1, max_len+1)])
# Total 8 columns: x, y, t, (e2, e3, e4, e5), (e6, e7, e8, e9)
data_node = pd.concat([data_node_grouped1[['x', 'y', 't']], e_cols1, e_cols2], axis=1)

save_path=config.source_path
with np.load(save_path, allow_pickle=True) as data:  
    t = data['t_eval']
    metadata = data['parameters'].item()
    data_node = bulid_aggr_feat(data,data_node)
data_node['e0']=data_node['e2']+data_node['e3']+data_node['e4']+data_node['e5']
data_node['e1']=data_node['e6']+data_node['e7']+data_node['e8']+data_node['e9']
data_node_x=data_node[['values_x', 'x', 'y', 't', 'ux', 'uy', 'uxx', 'uyy', 'x_prev', 'e0', 'e1']]
data_node_y=data_node[['values_y', 'x', 'y', 't', 'ux', 'uy', 'uxx', 'uyy', 'y_prev', 'e0', 'e1']]
print(data_node_x.head())
print(f'Aggregation features x: {data_node_x.shape}')
print(data_node_y.head())
print(f'Aggregation features y: {data_node_y.shape}')

# Data sampling
sample_size = 20000 
if len(data_msg_1) > sample_size:
    msg_sampled_1 = data_msg_1.sample(n=sample_size, random_state=42) # Set seed to ensure reproducibility
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

if len(data_node_y) > sample_size:
    node_sampled_y = data_node_y.sample(n=sample_size, random_state=42) 
    print(f"Sampling {sample_size} aggregate data points.")
else:
    node_sampled_y = data_node_y


# Step 2: Prepare data (X: independent variables, y: target e%d)
# msg1 candidates:  e0,  dx, dy, t, x_prev1,  y_prev1,  x_prev2,  y_prev2,  ux1,  ux2,  uy1,  uy2, uxx1, uxx2, uyy1, uyy2
# msg1_feature_names = ['t', 'y_prev1', 'y_prev2', 'x_prev1', 'x_prev2', 'ux1',  'ux2',  'uy1',  'uy2', 'uxx1', 'uxx2', 'uyy1', 'uyy2']
msg1_feature_names = ['t', 'y_prev1', 'y_prev2', 'x_prev1', 'x_prev2', 'ux1',  'ux2',  'uy1',  'uy2']

# msg2 candidates:  e1,  dx, dy, t, x_prev1,  y_prev1,  x_prev2,  y_prev2,  ux1,  ux2,  uy1,  uy2, uxx1, uxx2, uyy1, uyy2
# msg2_feature_names = ['t', 'y_prev1', 'y_prev2', 'x_prev1', 'x_prev2', 'ux1',  'ux2',  'uy1',  'uy2', 'uxx1', 'uxx2', 'uyy1', 'uyy2']
msg2_feature_names = ['t', 'y_prev1', 'y_prev2', 'x_prev1', 'x_prev2', 'ux1',  'ux2',  'uy1',  'uy2', ]
# nodeX candidates: values_x, x,  y,  t,  ux, uy, uxx, uyy, x_prev, e0, e1
node_x_feature_names = ['x', 'y', 't', 'ux','uy', 'x_prev', 'e0', 'e1']
# nodeY candidates: values_y, x,  y,  t,  ux, uy, uxx, uyy, y_prev, e0, e1
node_y_feature_names = ['x', 'y', 't', 'ux','uy', 'y_prev', 'e0', 'e1']

X_msg1 = msg_sampled_1[msg1_feature_names].values  # Shape: (n_samples, 5)
X_msg2 = msg_sampled_2[msg2_feature_names].values
y_msg1 = msg_sampled_1['e0'].values  # Shape: (n_samples,)
y_msg2 = msg_sampled_2['e1'].values

X_node_input = node_sampled_x[node_x_feature_names].values
Y_node_input = node_sampled_y[node_y_feature_names].values
X_node_output = node_sampled_x['values_x'].values
Y_node_output = node_sampled_y['values_y'].values


# Step 3: Initialize PySR regressor
# Customizable: niterations=more iterations, binary_operators=custom operators
SR_msg1 = PySRRegressor(
    niterations=100,           # Number of iterations (balance speed/accuracy)
    populations=50,
    population_size=50,
    parsimony=0.01,
    binary_operators=["+", "*", "-", "/"],  # Support addition, subtraction, multiplication, division
    unary_operators=["sin","cos"], # Optional unary operators
    model_selection="best",    # Select the best model
    output_directory='D:/MyProject/GNN+Nesy/SGN/result/msg_formula/msg1',
    random_state=42,
)
SR_msg2 = PySRRegressor(
    niterations=100,           # Number of iterations (balance speed/accuracy)
    populations=50,
    population_size=50,
    parsimony=0.01,
    binary_operators=["+", "*", "-", "/"],  # Support addition, subtraction, multiplication, division
    unary_operators=["sin","cos"], # Optional unary operators
    model_selection="best",    # Select the best model
    output_directory='D:/MyProject/GNN+Nesy/SGN/result/msg_formula/msg2',
)

SR_node_x = PySRRegressor(
    niterations=100,           # Number of iterations (balance speed/accuracy)
    populations=50,
    population_size=50,
    parsimony=0.01,
    binary_operators=["+", "*", "-", "/"],  # Support addition, subtraction, multiplication, division
    unary_operators=["sin","cos","exp","square"], # Optional unary operators
    model_selection="best",    # Select the best model
    output_directory='D:/MyProject/GNN+Nesy/SGN/result/node_formula/node_x',
    random_state=42,
)
SR_node_y = PySRRegressor(
    niterations=100,           # Number of iterations (balance speed/accuracy)
    populations=50,
    population_size=50,
    parsimony=0.01,
    binary_operators=["+", "*", "-", "/"],  # Support addition, subtraction, multiplication, division
    unary_operators=["sin","cos","exp","square"], # Optional unary operators
    model_selection="best",    # Select the best model
    output_directory='D:/MyProject/GNN+Nesy/SGN/result/node_formula/node_y',
    random_state=42,
)

# Step 4: Fit and output symbolic expression
SR_msg1.fit(X_msg1, y_msg1,variable_names=msg1_feature_names)
SR_msg2.fit(X_msg2, y_msg2,variable_names=msg2_feature_names)
SR_node_x.fit(X_node_input, X_node_output,variable_names=node_x_feature_names)
SR_node_y.fit(Y_node_input, Y_node_output,variable_names=node_y_feature_names)

# # Output best expression
formula_msg1 = SR_msg1.get_best()
loss_msg1= formula_msg1['loss']
print(f"Best message passing expression 1: {SR_msg1.sympy()}, Loss: {loss_msg1}")
formula_msg2 = SR_msg2.get_best()
loss_msg2= formula_msg2['loss']
print(f"Best message passing expression 2: {SR_msg2.sympy()}, Loss: {loss_msg2}")

formula_node_x = SR_node_x.get_best()
loss_node_x= formula_node_x['loss']
print(f"Best message aggregation expression x: {SR_node_x.sympy()}, Loss: {loss_node_x}")
formula_node_y = SR_node_y.get_best()
loss_node_y= formula_node_y['loss']
print(f"Best message aggregation expression y: {SR_node_y.sympy()}, Loss: {loss_node_y}")