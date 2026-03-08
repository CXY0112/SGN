############################################################
#   Tools required by the model, including visualization,  #
#   feature construction, message logging, etc.            #
############################################################

import numpy as np
import torch
import pandas as pd
import config
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

def build_feat_SG_2(result, window_length=11, polyorder=3):
    """
    Construct smooth features and labels from noisy results using a Savitzky-Golay filter:
    
    Args:
    - result: Data dictionary
    - window_length: Filter window length (must be odd)
    - polyorder: Fitting polynomial order, complexity
    
    X: (Nt, Nx*Ny, 8) -> [x, y, t, ux_smooth, uy_smooth, uxx_smooth, uyy_smooth, u_smooth(t-1)]
    Y: (Nt, Nx*Ny, 1) -> [u(t)] (Keep raw data as training target, or smooth as needed)
    """
    u = result['solution']    # (Nt+1, Ny, Nx)
    x_arr = result['x']       # (Nx,)
    y_arr = result['y']       # (Ny,)
    t_arr = result['t_eval']  # (Nt+1,)

    original_Nt_plus1, Ny, Nx = u.shape
    Nt = original_Nt_plus1 - 1
    NxNy = Nx * Ny

    # Calculate grid spacing (assuming uniform grid)
    dx = x_arr[1] - x_arr[0]
    dy = y_arr[1] - y_arr[0]

    # Generate spatial coordinate grid
    X, Y = np.meshgrid(x_arr, y_arr, indexing='xy')  # (Ny, Nx)

    # Initialize containers
    X_feat = np.zeros((Nt, NxNy, 8))
    Y_feat = np.zeros((Nt, NxNy, 1))

    # Iterate through time steps, starting from i=1
    for i in range(1, original_Nt_plus1):
        t = t_arr[i]  # Current t
        
        # 1. Basic coordinate features [x, y, t]
        xy_t = np.stack([X.flatten(), Y.flatten(), np.full(X.size, t)], axis=1) # (Nx*Ny, 3)
        
        # 2. Get u from the previous time step
        u_prev_2d = u[i-1]  # (Ny, Nx) - This is the noisy raw data
        
        # 3. Calculate 1st and 2nd order partial derivatives - using S-G filter
        # axis=1 corresponds to x-axis (columns), deriv=1 for 1st derivative, delta=dx to correct values
        ux_2d_smooth = savgol_filter(u_prev_2d, window_length, polyorder, deriv=1, delta=dx, axis=1)
        uxx_2d_smooth = savgol_filter(u_prev_2d, window_length, polyorder, deriv=2, delta=dx, axis=1)
        
        # 4. Calculate 1st and 2nd order partial derivatives - using S-G filter
        # axis=0 corresponds to y-axis (rows), deriv=1 for 1st derivative, delta=dy to correct values
        uy_2d_smooth = savgol_filter(u_prev_2d, window_length, polyorder, deriv=1, delta=dy, axis=0)
        uyy_2d_smooth = savgol_filter(u_prev_2d, window_length, polyorder, deriv=2, delta=dy, axis=0)
        
        # 5. Calculate smoothed u (u_smooth)
        u_temp = savgol_filter(u_prev_2d, window_length, polyorder, deriv=0, axis=1)
        u_prev_2d_smooth = savgol_filter(u_temp, window_length, polyorder, deriv=0, axis=0)

        # Flatten
        ux = ux_2d_smooth.flatten()       # (Nx*Ny,)
        uy = uy_2d_smooth.flatten()       # (Nx*Ny,)
        uxx = uxx_2d_smooth.flatten()     # (Nx*Ny,)
        uyy = uyy_2d_smooth.flatten()     # (Nx*Ny,)
        prev_u = u_prev_2d_smooth.flatten() # (Nx*Ny,) - Use smoothed u
        
        # ------------------- Core modification ends -------------------

        # 6. Combine into [x, y, t, ux_smooth, uy_smooth, uxx_smooth, uyy_smooth, u_smooth]
        X_feat[i-1] = np.column_stack([
            xy_t, 
            ux.reshape(-1, 1), 
            uy.reshape(-1, 1), 
            uxx.reshape(-1, 1),
            uyy.reshape(-1, 1),
            prev_u.reshape(-1, 1)
        ])
        
        # 7. Label Y
        # Typically, raw data is used for labels, allowing the network to learn to approximate 
        # the true distribution (the center of the noisy distribution)
        Y_feat[i-1] = u[i].reshape(-1, 1)

    return X_feat, Y_feat

def build_feat_SG(result, window_length=11, polyorder=3):
    """
    Construct smooth features and labels from noisy results using a Savitzky-Golay filter:
    
    Args:
    - result: Data dictionary
    - window_length: Filter window length (must be odd)
    - polyorder: Fitting polynomial order, complexity
    
    X: (Nt, Nx*Ny, 6) -> [x, y, t, ux_smooth, uy_smooth, u_smooth(t-1)]
    Y: (Nt, Nx*Ny, 1) -> [u(t)] (Keep raw data as training target, or smooth as needed)
    """
    u = result['solution']    # (Nt+1, Ny, Nx)
    x_arr = result['x']       # (Nx,)
    y_arr = result['y']       # (Ny,)
    t_arr = result['t_eval']  # (Nt+1,)

    original_Nt_plus1, Ny, Nx = u.shape
    Nt = original_Nt_plus1 - 1
    NxNy = Nx * Ny

    # Calculate grid spacing (assuming uniform grid)
    dx = x_arr[1] - x_arr[0]
    dy = y_arr[1] - y_arr[0]

    # Generate spatial coordinate grid
    X, Y = np.meshgrid(x_arr, y_arr, indexing='xy')  # (Ny, Nx)

    # Initialize containers
    X_feat = np.zeros((Nt, NxNy, 6))
    Y_feat = np.zeros((Nt, NxNy, 1))

    # Iterate through time steps, starting from i=1
    for i in range(1, original_Nt_plus1):
        t = t_arr[i]  # Current t
        
        # 1. Basic coordinate features [x, y, t]
        xy_t = np.stack([X.flatten(), Y.flatten(), np.full(X.size, t)], axis=1) # (Nx*Ny, 3)
        
        # 2. Get u from the previous time step
        u_prev_2d = u[i-1]  # (Ny, Nx) - This is the noisy raw data

        # ------------------- Core modification starts -------------------
        
        # 3. Calculate 1st order partial derivative ux (∂u/∂x) - using S-G filter
        # axis=1 corresponds to x-axis (columns), deriv=1 for 1st derivative, delta=dx to correct values
        ux_2d_smooth = savgol_filter(u_prev_2d, window_length, polyorder, deriv=1, delta=dx, axis=1)
        
        # 4. Calculate 1st order partial derivative uy (∂u/∂y) - using S-G filter
        # axis=0 corresponds to y-axis (rows), deriv=1 for 1st derivative, delta=dy to correct values
        uy_2d_smooth = savgol_filter(u_prev_2d, window_length, polyorder, deriv=1, delta=dy, axis=0)

        # 5. Calculate smoothed u (u_smooth)
        # It is recommended to also denoise u in the input features, otherwise the network will 
        # see clean gradients but noisy values, which can easily cause confusion.
        # Strategy: smooth along x first, then along y (simple 2D smoothing approximation)
        u_temp = savgol_filter(u_prev_2d, window_length, polyorder, deriv=0, axis=1)
        u_prev_2d_smooth = savgol_filter(u_temp, window_length, polyorder, deriv=0, axis=0)

        # Flatten
        ux = ux_2d_smooth.flatten()       # (Nx*Ny,)
        uy = uy_2d_smooth.flatten()       # (Nx*Ny,)
        prev_u = u_prev_2d_smooth.flatten() # (Nx*Ny,) - Use smoothed u
        
        # ------------------- Core modification ends -------------------

        # 6. Combine into [x, y, t, ux_smooth, uy_smooth, u_smooth]
        X_feat[i-1] = np.column_stack([
            xy_t, 
            ux.reshape(-1, 1), 
            uy.reshape(-1, 1), 
            prev_u.reshape(-1, 1)
        ])
        
        # 7. Label Y
        # Typically, raw data is used for labels, allowing the network to learn to approximate 
        # the true distribution (the center of the noisy distribution)
        Y_feat[i-1] = u[i].reshape(-1, 1)

    return X_feat, Y_feat

def bulid_feat_2(result):
    """
    Construct features and labels from results:
    X: (Nt, Nx*Ny, 7) -> [x, y, t, ux, uy, x_solution(x, y, t-1), y_solution(x, y, t-1)]  # Exclude initial step, start from i=1
    Y: (Nt, Nx*Ny, 2) -> [x_solution(x, y, t), y_solution(x, y, t)]
    """

    # u = result['solution']    # (Nt+1, Ny, Nx)
    x_sol = result['x_solution']  # (Nt+1, Ny, Nx)
    y_sol = result['y_solution']  # (Nt+1, Ny, Nx)
    x_arr = result['x']       # (Nx,)
    y_arr = result['y']       # (Ny,)
    t_arr = result['t_eval']  # (Nt+1,)

    original_Nt_plus1, Ny, Nx = x_sol.shape
    Nt = original_Nt_plus1 - 1
    NxNy = Nx * Ny

    # Generate spatial coordinate grid
    X, Y = np.meshgrid(x_arr, y_arr, indexing='xy')  # (Ny, Nx)

    # Initialize containers
    X_feat = np.zeros((Nt, NxNy, 7))
    Y_feat = np.zeros((Nt, NxNy, 2))

    # Iterate through time steps, starting from i=1
    for i in range(1, original_Nt_plus1):
        t = t_arr[i]  # Current t
        # Spatial features for each time step [x, y, t]
        xy_t = np.stack([X.flatten(), Y.flatten(), np.full(X.size, t)], axis=1)  # (Nx*Ny, 3)
        
        # u value from the previous time step: u[i-1], keep as 2D for gradient calculation
        x_prev_2d = x_sol[i-1]  # (Ny, Nx)
        y_prev_2d = y_sol[i-1]  # (Ny, Nx)
        prev_x = x_prev_2d.flatten()  # (Nx*Ny,)
        prev_y = y_prev_2d.flatten()  # (Nx*Ny,)

        # Calculate ux = ∂(x_solution(t-1))/∂x and uy = ∂(y_solution(t-1))/∂y
        ux_2d = np.gradient(x_prev_2d, x_arr, axis=1)  # (Ny, Nx), along x direction
        uy_2d = np.gradient(y_prev_2d, y_arr, axis=0)  # (Ny, Nx), along y direction
        ux = ux_2d.flatten()  # (Nx*Ny,)
        uy = uy_2d.flatten()  # (Nx*Ny,)

        # Combine into [x, y, t, ux, uy, x_prev, y_prev]
        X_feat[i-1] = np.column_stack([
            xy_t, 
            ux.reshape(-1, 1), 
            uy.reshape(-1, 1), 
            prev_x.reshape(-1, 1), 
            prev_y.reshape(-1, 1)
        ])  # (Nx*Ny, 7)
        
        # Current solution as label [x_sol(t), y_sol(t)]
        Y_feat[i-1] = np.column_stack([
            x_sol[i].flatten(), 
            y_sol[i].flatten()
        ])

    return X_feat, Y_feat

def build_feat_2_SG(result, window_length=5, polyorder=3):
    """
    For bivariate systems (e.g., Navier-Stokes u,v or Gray-Scott U,V),
    construct smooth features and labels using a Savitzky-Golay filter.

    X: (Nt, Nx*Ny, 9) -> [x, y, t, ux_smooth, uy_smooth, uxx, uyy, x_sol_smooth(t-1), y_sol_smooth(t-1)]
    Y: (Nt, Nx*Ny, 2) -> [x_solution(t), y_solution(t)]
    """

    x_sol = result['x_solution']  # (Nt+1, Ny, Nx) - Corresponds to physical field u
    y_sol = result['y_solution']  # (Nt+1, Ny, Nx) - Corresponds to physical field v
    x_arr = result['x']       # (Nx,)
    y_arr = result['y']       # (Ny,)
    t_arr = result['t_eval']  # (Nt+1,)

    original_Nt_plus1, Ny, Nx = x_sol.shape
    Nt = original_Nt_plus1 - 1
    NxNy = Nx * Ny

    # Calculate grid spacing (for correct derivative calculation)
    dx = x_arr[1] - x_arr[0]
    dy = y_arr[1] - y_arr[0]

    # Generate spatial coordinate grid
    X, Y = np.meshgrid(x_arr, y_arr, indexing='xy')  # (Ny, Nx)

    # Initialize containers
    X_feat = np.zeros((Nt, NxNy, 9))
    Y_feat = np.zeros((Nt, NxNy, 2))

    # Iterate through time steps, starting from i=1
    for i in range(1, original_Nt_plus1):
        t = t_arr[i]  # Current t
        
        # 1. Basic coordinate features [x, y, t]
        xy_t = np.stack([X.flatten(), Y.flatten(), np.full(X.size, t)], axis=1) # (Nx*Ny, 3)
        
        # 2. Get field values from the previous time step (noisy)
        x_prev_2d = x_sol[i-1]  # (Ny, Nx)
        y_prev_2d = y_sol[i-1]  # (Ny, Nx)

        # ------------------- Core modification -------------------

        # Smooth x_solution (i.e., u)
        x_temp = savgol_filter(x_prev_2d, window_length, polyorder, deriv=0, axis=1)
        x_prev_smooth = savgol_filter(x_temp, window_length, polyorder, deriv=0, axis=0)
        
        # Smooth y_solution (i.e., v)
        y_temp = savgol_filter(y_prev_2d, window_length, polyorder, deriv=0, axis=1)
        y_prev_smooth = savgol_filter(y_temp, window_length, polyorder, deriv=0, axis=0)

        # 4. Robust differentiation
        # Note: Original code logic is x_sol derivative with respect to x, y_sol derivative with respect to y
        
        # Calculate ux = ∂(x_solution)/∂x
        ux_2d_smooth = savgol_filter(x_prev_2d, window_length, polyorder, deriv=1, delta=dx, axis=1)
        uxx_2d_smooth = savgol_filter(x_prev_2d, window_length, polyorder, deriv=2, delta=dx, axis=1)

        # Calculate uy = ∂(y_solution)/∂y
        uy_2d_smooth = savgol_filter(y_prev_2d, window_length, polyorder, deriv=1, delta=dy, axis=0)
        uyy_2d_smooth = savgol_filter(y_prev_2d, window_length, polyorder, deriv=2, delta=dy, axis=0)
        
        # ------------------- Core modification ends -------------------

        # Flatten data
        ux = ux_2d_smooth.flatten()
        uy = uy_2d_smooth.flatten()
        uxx = uxx_2d_smooth.flatten()
        uyy = uyy_2d_smooth.flatten()
        prev_x_flat = x_prev_smooth.flatten()
        prev_y_flat = y_prev_smooth.flatten()

        # 5. Combine features
        # [x, y, t, ∂x_sol/∂x, ∂y_sol/∂y, x_sol, y_sol]
        X_feat[i-1] = np.column_stack([
            xy_t, 
            ux.reshape(-1, 1), 
            uy.reshape(-1, 1), 
            uxx.reshape(-1, 1),
            uyy.reshape(-1, 1),
            prev_x_flat.reshape(-1, 1), 
            prev_y_flat.reshape(-1, 1)
        ])
        
        # 6. Label Y
        # Labels typically keep original observations (including noise), 
        # forcing the network to learn the denoised manifold
        Y_feat[i-1] = np.column_stack([
            x_sol[i].flatten(), 
            y_sol[i].flatten()
        ])

    return X_feat, Y_feat

def build_feat_3(result, if_gauss=False, sigma_smooth=0.8):
    """
    Construct features and labels from Burgers' equation solution results.
    Features include coordinates, time, current state values, and 1st/2nd order derivatives.

    X: (Nt, Nx*Ny, 9) -> [x, y, t, ux, uy, uxx, uyy, x_sol(t-1), y_sol(t-1)]
    Y: (Nt, Nx*Ny, 2) -> [x_sol(t), y_sol(t)]

    Args:
    - result: dict, dictionary containing solution results 'x_solution', 'y_solution', 'x', 'y', 't_eval'.
    - if_gauss: bool, if True, apply Gaussian smoothing to reduce noise before calculating partial derivatives.
    - sigma_smooth: float, standard deviation of the Gaussian kernel. Larger value means stronger smoothing.

    Returns:
    - X_feat: np.ndarray, feature matrix.
    - Y_feat: np.ndarray, label matrix.
    """

    # 1. Unpack data
    x_sol = result['x_solution']      # (Nt+1, Ny, Nx) - Raw data containing noise
    y_sol = result['y_solution']      # (Nt+1, Ny, Nx)
    x_arr = result['x']               # (Nx,)
    y_arr = result['y']               # (Ny,)
    t_arr = result['t_eval']          # (Nt+1,)

    original_Nt_plus1, Ny, Nx = x_sol.shape
    Nt = original_Nt_plus1 - 1
    NxNy = Nx * Ny

    # 2. Generate spatial coordinate grid
    # 'xy' indexing ensures X varies along columns (x-axis), Y varies along rows (y-axis)
    X, Y = np.meshgrid(x_arr, y_arr, indexing='xy')  # (Ny, Nx)

    # 3. Initialize containers (Feature dimension 9)
    X_feat = np.zeros((Nt, NxNy, 9))
    Y_feat = np.zeros((Nt, NxNy, 2))

    # 4. Iterate through time steps, construct mapping from (t-1) -> (t)
    for i in range(1, original_Nt_plus1):
        t = t_arr[i]  # Current target prediction time t
        
        # State at previous time step (t-1)
        x_prev_2d_raw = x_sol[i-1]
        y_prev_2d_raw = y_sol[i-1]
        
        # --- Core logic: Smoothing processing ---
        if if_gauss:
            # Enable Gaussian smoothing to reduce noise amplification in derivative calculations
            x_source = gaussian_filter(x_prev_2d_raw, sigma=sigma_smooth)
            y_source = gaussian_filter(y_prev_2d_raw, sigma=sigma_smooth)
        else:
            # Disable Gaussian smoothing, directly use raw (noisy) data for finite differences
            x_source = x_prev_2d_raw
            y_source = y_prev_2d_raw

        # --- Feature preparation ---
        # Basic features [x, y, t]
        xy_t = np.stack([X.flatten(), Y.flatten(), np.full(X.size, t)], axis=1)
        
        # Current state features (use source data used for calculating derivatives as features)
        prev_x = x_source.flatten()
        prev_y = y_source.flatten()

        # --- Derivative calculations (using x_source and y_source) ---
        
        # 1. First-order derivatives
        # ux = ∂(x_sol)/∂x, uy = ∂(y_sol)/∂y
        ux_2d = np.gradient(x_source, x_arr, axis=1)    # Along x-axis
        uy_2d = np.gradient(y_source, y_arr, axis=0)    # Along y-axis
        
        # 2. Second-order derivatives
        # uxx = ∂²(x_sol)/∂x², uyy = ∂²(y_sol)/∂y²
        # Differentiate the first-order derivative again
        uxx_2d = np.gradient(ux_2d, x_arr, axis=1)      # Along x-axis again
        uyy_2d = np.gradient(uy_2d, y_arr, axis=0)      # Along y-axis again

        # Flatten derivative features
        ux = ux_2d.flatten()
        uy = uy_2d.flatten()
        uxx = uxx_2d.flatten()
        uyy = uyy_2d.flatten()

        # --- Feature combination ---
        # Order: [x, y, t, ux, uy, uxx, uyy, prev_x, prev_y]
        X_feat[i-1] = np.column_stack([
            xy_t,                   # 0,1,2: Coordinates and time
            ux.reshape(-1, 1),      # 3: 1st order derivative x
            uy.reshape(-1, 1),      # 4: 1st order derivative y
            uxx.reshape(-1, 1),     # 5: 2nd order derivative xx
            uyy.reshape(-1, 1),     # 6: 2nd order derivative yy
            prev_x.reshape(-1, 1),  # 7: Current state x used for derivative calculation
            prev_y.reshape(-1, 1)   # 8: Current state y used for derivative calculation
        ])
        
        # --- Labels ---
        # Label Y is always the raw (noisy) data of the next time step, which is the prediction target of the model
        Y_feat[i-1] = np.column_stack([
            x_sol[i].flatten(), 
            y_sol[i].flatten()
        ])

    return X_feat, Y_feat

# Used to record messages for subsequent PySR formula fitting

def get_messages(pgn,loss_type,msg_dim,newtestloader,dim=2):

    def get_message_info(tmp,dim=2):
        pgn.cpu()

        s1 = tmp.x[tmp.edge_index[0]]
        s2 = tmp.x[tmp.edge_index[1]]
        tmp = torch.cat([s1, s2], dim=1)  # tmp has shape [E, 2 * in_channels]
        tmp = tmp.float()
        if loss_type == '_kl_':
            raw_msg = pgn.msg_fnc(tmp)
            mu = raw_msg[:, 0::2]
            logvar = raw_msg[:, 1::2]
            m12 = mu
        else:
            m12 = pgn.msg_fnc(tmp)

        all_messages = torch.cat((
            s1,
            s2,
            m12), dim=1)

        columns = [elem%(k) for k in range(1, 3) for elem in config.columns.split(' ')]
        columns += ['e%d'%(k,) for k in range(msg_dim)]

        return pd.DataFrame(
            data=all_messages.cpu().detach().numpy(),
            columns=columns
        )

    msg_info = []
    for i, g in enumerate(newtestloader):
        msg_info.append(get_message_info(g))

    msg_info = pd.concat(msg_info)
    
    if dim == 2:
        msg_info['dx'] = msg_info.x1 - msg_info.x2
        msg_info['dy'] = msg_info.y1 - msg_info.y2
    if dim == 3:
        msg_info['dx'] = msg_info.x1 - msg_info.x2
        msg_info['dy'] = msg_info.y1 - msg_info.y2
        msg_info['dz'] = msg_info.z1 - msg_info.z2
    
    return msg_info

def new_loss(self, g, loss_type, batch, n, square=False):
    if square:
        return torch.sum((g.y - self.just_derivative(g))**2)
    else:
        base_loss = torch.sum(torch.abs(g.y - self.just_derivative(g)))
        s1 = g.x[self.edge_index[0]]
        s2 = g.x[self.edge_index[1]]
        if loss_type == '_l1_':
            m12 = self.message(s1, s2)
            regularization = 1e-2
            normalized_l05 = torch.sum(torch.abs(m12)).to(dtype=base_loss.dtype)
            return base_loss, regularization * batch * normalized_l05 / n**2 * n
        return base_loss

def m_loss(self, g, loss_type, batch, n):
    # base_loss = torch.sum(torch.abs(g.y - self.just_derivative(g)))
    # Extract u1, v1, m1 from g.y
    u1 = g.y[..., 0]  # Assuming the last dimension is [u1, v1, m1]
    v1 = g.y[..., 1]
    # Extract u, v, m from just_derivative(g)
    deriv = self.just_derivative(g)
    u = deriv[..., 0]   # Assuming the last dimension is [u, v, m]
    v = deriv[..., 1]
    # Calculate absolute difference for each part
    loss1 = torch.abs(u1 - u)
    loss2 = torch.abs(v1 - v)
    loss3 = torch.abs(
        (torch.square(u1) + torch.square(v1)) - 
        (torch.square(u) + torch.square(v))
    )
    # Total loss: sum after element-wise addition
    base_loss = torch.sum(loss1 + loss2 + 0.2*loss3)

    s1 = g.x[self.edge_index[0]]
    s2 = g.x[self.edge_index[1]]
    if loss_type in ['_l1_', 'momentum']:
        m12 = self.message(s1, s2)
        regularization = 1e-2
        normalized_l05 = torch.sum(torch.abs(m12)).to(dtype=base_loss.dtype)
        return base_loss, regularization * batch * normalized_l05 / n**2 * n
    return base_loss

# Visualize 2D data
def create_interactive_wave_viz(u_analytical, t_eval, Lx=1.0, Ly=1.0, interval=0.05,vmin=None, vmax=None):
    """
    Creates an interactive API for 2D wave equation visualization.
    
    Args:
    - u_analytical: np.ndarray, shape (Nt, Ny, Nx), spatio-temporal wave field data.
    - t_eval: np.ndarray, shape (Nt,), time array.
    - Lx, Ly: float, spatial domain size, default 1.0.
    - interval: float, playback interval in seconds, default 0.05 (~50ms/frame).
    
    Returns: None, directly displays the interactive window.
    """
    Nt, Ny, Nx = u_analytical.shape
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)  # Leave space for controls

    # Automatically calculate color range (if not specified)
    if vmin is None:
        vmin = float(np.min(u_analytical))
    if vmax is None:
        vmax = float(np.max(u_analytical))

    # Playback control variables
    playing = False

    # Initial plot function
    def init_plot(frame_idx):
        ax.clear()
        im = ax.imshow(u_analytical[frame_idx], extent=[0, Lx, 0, Ly], origin='lower',
                       cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f't = {t_eval[frame_idx]:.2f}')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        # Add colorbar (only once)
        if not hasattr(init_plot, 'cbar'):
            init_plot.cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        return im

    # Initial frame
    init_frame = 0
    im = init_plot(init_frame)

    # Add Slider
    ax_slider = plt.axes([0.15, 0.12, 0.65, 0.03])
    slider = Slider(ax_slider, 'Frame (t)', 0, Nt - 1, valinit=init_frame, valstep=1)

    # Update function
    def update(val):
        frame_idx = int(slider.val)
        init_plot(frame_idx)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Play function
    def play(event):
        nonlocal playing
        if not playing:
            playing = True
            current_frame = int(slider.val)
            while playing and current_frame < Nt - 1:
                current_frame += 1
                slider.set_val(current_frame)
                update(None)
                plt.pause(interval)
            playing = False

    # Pause function
    def pause(event):
        nonlocal playing
        playing = False

    # Add buttons
    ax_play = plt.axes([0.15, 0.05, 0.1, 0.04])
    play_button = Button(ax_play, 'Play')
    play_button.on_clicked(play)

    ax_pause = plt.axes([0.35, 0.05, 0.1, 0.04])
    pause_button = Button(ax_pause, 'Pause')
    pause_button.on_clicked(pause)

    plt.show(block=True)  # Block display to ensure window stays open


def create_interactive_comparison_viz(train_path,
                                      pred_path,
                                      train_y='solution',
                                      pred_y='solution',
                                    interval=0.05):
    """
    Create interactive comparison visualization
    """
    # Load training data
    try:
        with np.load(train_path, allow_pickle=True) as data:
            train_values = data[train_y][1:]  # Skip t=0, shape (100, 128, 128)
            t_eval = data['t_eval'][1:]  # Time array, starting from t=1
            train_metadata = data['parameters'].item()
    except Exception as e:
        print(f"Failed to load training data: {e}")
        return
    
    # Load prediction data
    try:
        with np.load(pred_path, allow_pickle=True) as data:
            pred_values = data[pred_y]  # Shape (100, 128, 128)
            # If prediction data has t_eval, use it; otherwise reuse training's
            if 't_eval' in data:
                pred_t = data['t_eval']
                if (len(pred_t)-1) != len(t_eval):
                    print("Warning: Prediction t_eval does not match training, using training t_eval")
                    pred_t = t_eval
            else:
                pred_t = t_eval
            pred_metadata = data['parameters'].item() if 'parameters' in data else {}
    except Exception as e:
        print(f"Failed to load prediction data: {e}")
        return
    
    # Update Lx, Ly from metadata (prioritize training data)
    Lx = train_metadata.get('Lx', pred_metadata.get('Lx', 1))
    Ly = train_metadata.get('Ly', pred_metadata.get('Ly', 1))
    
    Nt = train_values.shape[0]  # Should be 100
    Ny, Nx = train_values.shape[1:]  # 128, 128
    
    # Check shape match
    if pred_values.shape != (Nt, Ny, Nx):
        print(f"Shape mismatch: Training {train_values.shape}, Prediction {pred_values.shape}")
        return
    
    plt.ion()  # Enable interactive mode
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))  # No constrained_layout
    # Fixed layout spacing (called only once)
    plt.subplots_adjust(left=0.05, right=0.92, top=0.85, bottom=0.25, wspace=0.3)
    
    # Global color range
    vmin = float(np.min([np.min(train_values), np.min(pred_values)]))
    vmax = float(np.max([np.max(train_values), np.max(pred_values)]))
    
    # Playback control
    playing = False
    
    # Initial plot function (does not adjust layout, only updates content)
    def init_plot(frame_idx):
        # Clear subplots
        ax_left.clear()
        ax_right.clear()
        
        # Plot training data (left)
        im_left = ax_left.imshow(train_values[frame_idx], extent=[0, Lx, 0, Ly], origin='lower',
                                 cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax_left.set_title('Ground Truth', fontsize=12, pad=10)
        ax_left.set_xlabel('$x$')
        ax_left.set_ylabel('$y$')
        
        # Plot prediction data (right)
        im_right = ax_right.imshow(pred_values[frame_idx], extent=[0, Lx, 0, Ly], origin='lower',
                                   cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax_right.set_title('Prediction', fontsize=12, pad=10)
        ax_right.set_xlabel('$x$')
        
        # Time title (top center)
        fig.suptitle(f't = {t_eval[frame_idx]:.2f}', fontsize=14, y=0.92)
        
        # Add colorbar only once (moved outside function, but controlled by flag)
        if not hasattr(init_plot, 'cbar_added'):
            cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # Move right slightly to avoid squeezing
            init_plot.cbar = fig.colorbar(im_left, cax=cbar_ax, shrink=0.8)
            init_plot.cbar.set_label('Value', rotation=270, labelpad=15)
            init_plot.cbar_added = True
        
        return im_left, im_right
    
    # Initial frame
    init_frame = 0
    im_left, im_right = init_plot(init_frame)
    
    # Add Slider
    ax_slider = plt.axes([0.15, 0.08, 0.65, 0.03])
    slider = Slider(ax_slider, 'Frame Index', 0, Nt - 1, valinit=init_frame, valstep=1)
    
    # Update function (only updates images)
    def update(val):
        frame_idx = int(slider.val)
        init_plot(frame_idx)
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Play function
    def play(event):
        nonlocal playing
        if not playing:
            playing = True
            current_frame = int(slider.val)
            while playing and current_frame < Nt - 1:
                current_frame += 1
                slider.set_val(current_frame)
                update(None)
                plt.pause(interval)
            playing = False
    
    # Pause function
    def pause(event):
        nonlocal playing
        playing = False
    
    # Add buttons
    ax_play = plt.axes([0.15, 0.02, 0.08, 0.04])
    play_button = Button(ax_play, 'Play')
    play_button.on_clicked(play)
    
    ax_pause = plt.axes([0.28, 0.02, 0.08, 0.04])
    pause_button = Button(ax_pause, 'Pause')
    pause_button.on_clicked(pause)
    
    plt.show(block=True)


def inject_noise(x, noise_level=0.01):
    """
    Perform dynamic noise injection on input features x.
    
    Args:
    - x: Input feature tensor (Batch, Features)
    - noise_level: Noise intensity relative to x standard deviation (default 1%)
    
    Returns:
    - x_noisy: Feature with added noise
    """
    if not x.is_floating_point():
        x = x.float()
    
    # Calculate standard deviation of current batch for adaptive noise amplitude
    # detach() prevents gradient backpropagation to std calculation
    std = x.std(dim=0, keepdim=True).detach()
    
    # Generate standard normal distribution noise N(0, 1)
    noise = torch.randn_like(x)
    
    # Scale noise: noise = level * std * N(0, 1)
    # Add a small epsilon 1e-6 to prevent std from being 0
    scaled_noise = noise * (std * noise_level + 1e-6)
    
    return x + scaled_noise

def estimate_noise_level(data_u,window_length=15, polyorder=3):
    """
    Estimate noise level of data
    data_u: Raw noisy data (Nt, Ny, Nx)
    """
    from scipy.signal import savgol_filter
    
    # 1. Select a slice or randomly select part of the data
    sample = data_u[10] # Take the 10th frame
    
    # 2. Perform strong smoothing (as an approximation of Ground Truth)
    # Larger window to ensure most noise is filtered out
    smooth = savgol_filter(sample, window_length=window_length, polyorder=polyorder, axis=0)
    smooth = savgol_filter(smooth, window_length=window_length, polyorder=polyorder, axis=1)
    
    # 3. Calculate residual (Noise estimation)
    residual = sample - smooth
    
    # 4. Calculate signal-to-noise ratio (Noise Level = Std(Noise) / Std(Signal))
    signal_std = np.std(sample)
    noise_std = np.std(residual)
    
    estimated_level = noise_std / signal_std
    return estimated_level