####################################
#    Data Generation for Training  #
####################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from torch import nn  # For Conv2d Laplacian
import torch  # New: GPU acceleration
import os
import time

# Visualize 2D data
def create_interactive_wave_viz(u_analytical, t_eval, Lx=1.0, Ly=1.0, interval=0.05, vmin=None, vmax=None):
    """
    Creates an interactive API for 2D wave equation visualization.
    
    Parameters:
    - u_analytical: np.ndarray, shape (Nt, Ny, Nx), spatio-temporal wave field data.
    - t_eval: np.ndarray, shape (Nt,), time array.
    - Lx, Ly: float, spatial domain size, default 1.0.
    - interval: float, playback interval (seconds), default 0.05 (~50ms/frame).
    
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

def generate_2d_wave_analytical(Lx=1.0, Ly=1.0, t0=0.0, t_end=2.0, Nx=128, Ny=128, Nt=100, c=1.0, save_path=None):
    """
    Generates numerical data for the analytical fundamental mode solution of the 2D wave equation.

    This function calculates the analytical fundamental mode solution for the 2D wave equation on a 
    rectangular domain [0, Lx] × [0, Ly] with Dirichlet boundary conditions (u=0 on boundaries). 
    The solution is obtained via the separation of variables method and represents a standing wave.

    Analytical formula:

    $$ u(x, y, t) = \sin\left( \frac{\pi x}{L_x} \right) \sin\left( \frac{\pi y}{L_y} \right) \cos\left( \frac{\pi c t}{L_x} \right) $$

    Where:
    - The spatial part ensures zero boundaries, forming a standing wave pattern.
    - The temporal part is a harmonic oscillation with frequency determined by $c$ and $L_x$ (period $T = 2 L_x / c$).

    Parameters:
    - Lx, Ly: float, spatial domain length [0, Lx] x [0, Ly] (default 1.0).
    - t0, t_end: float, start and end points of the time interval (default 0, 2).
    - Nx, Ny: int, number of spatial discretization points (default 128).
    - Nt: int, number of time discretization points (default 100).
    - c: float, wave speed (default 1.0).
    - save_path: str, optional NPZ save path (default None, no saving).

    Returns:
    - dict: {'solution': u_analytical (Nt, Ny, Nx), 'x': x_array, 'y': y_array, 't_eval': t_array}
    """
    # Generate grid
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    t_eval = np.linspace(t0, t_end, Nt+1)
    X, Y = np.meshgrid(x, y)  # 2D spatial grid

    # Define analytical solution: Separation of variables fundamental mode
    def analytical_solution(X, Y, t, c, Lx, Ly):
        return np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly) * np.cos(np.pi * c * t / Lx)

    # Calculate analytical solution (shape: (Nt, Ny, Nx))
    u_analytical = np.zeros((Nt+1, Ny, Nx))
    for i, t in enumerate(t_eval):
        u_analytical[i] = analytical_solution(X, Y, t, c, Lx, Ly)

    # Assemble output dictionary
    result = {
        'solution': u_analytical,
        'x': x,
        'y': y,
        't_eval': t_eval,
        'parameters': {
            'Lx': Lx,
            'Ly': Ly,
            't0': t0,
            't_end': t_end,
            'Nx': Nx,
            'Ny': Ny,
            'Nt': Nt,
            'c': c
        }
    }

    # Optional: Save to NPZ file
    if save_path is not None:
        np.savez(save_path, **result)
        print(f"Results saved to {save_path}")

    print(f"Solution shape: {u_analytical.shape}")
    return result

def generate_2d_naviers_stokes(Lx=2 * np.pi, Ly=2 * np.pi, t0=0.0, t_end=10.0, Nx=128, Ny=128, Nt=100, 
                               nu=0.01, U_max=1.0, save_path=None, seed=42):
    """
    Solves the 2D incompressible Navier-Stokes equations using the projection method (PyTorch/CUDA accelerated),
    with Taylor-Green vortex as the initial condition, to generate decaying vortex flow data under periodic boundaries.

    Mathematical Model (2D Incompressible N-S):
    $$ \nabla \cdot \mathbf{u} = 0 $$
    $$ \partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u} $$

    Numerical Method: Chorin Projection Method (low-order variant), explicit Euler time-stepping, 
    second-order central difference + FFT Poisson solver.
    Boundaries: Periodic. This fixed version ensures energy conservation/decay physics are reasonable.

    Parameters: (As before)
    Returns: dict with velocity_solution (Nt+1, 2, Ny, Nx), etc.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    integration_factor = 200  # Refine time step to improve stability
    num_steps = Nt * integration_factor
    dt = (t_end - t0) / num_steps

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    dx = Lx / Nx
    dy = Ly / Ny
    
    t_eval = np.linspace(t0, t_end, Nt + 1) 

    # Enhance CFL (Conservative factor 0.4)
    cfl_adv = 0.4 * min(dx, dy) / U_max
    cfl_diff = 0.4 * min(dx**2, dy**2) / nu
    cfl_limit = min(cfl_adv, cfl_diff)
    if dt > cfl_limit:
        print(f"⚠️ Warning: CFL suggests dt < {cfl_limit:.6f}, currently {dt:.6f}. Suggest increasing integration_factor.")
    print(f"dt: {dt:.6f}, Total steps: {num_steps}")

    X, Y = np.meshgrid(x, y, indexing='ij')
    u0_np = np.cos(X) * np.sin(Y)
    v0_np = -np.sin(X) * np.cos(Y)
    
    velocity = torch.tensor(np.stack([u0_np, v0_np], axis=0), dtype=torch.float32, device=device)[None, :, :, :]

    vel_traj = []
    sample_indices = np.round(np.linspace(0, num_steps, Nt + 1, endpoint=True)).astype(int)
    sample_index_set = set(sample_indices)

    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    ksq = KX**2 + KY**2
    ksq[0, 0] = 1e-12
    ksq_torch = torch.tensor(ksq, dtype=torch.float32, device=device)

    for step in range(num_steps + 1):
        if step in sample_index_set:
            vel_traj.append(velocity.squeeze(0).cpu().numpy())
        
        if step == num_steps:
            break

        u = velocity[0, 0]
        v = velocity[0, 1]

        def grad_x(f):
            return (torch.roll(f, -1, dims=1) - torch.roll(f, 1, dims=1)) / (2 * dx)
        def grad_y(f):
            return (torch.roll(f, -1, dims=0) - torch.roll(f, 1, dims=0)) / (2 * dy)
        def lap(f):
            lap_x = (torch.roll(f, 1, dims=1) + torch.roll(f, -1, dims=1) - 2 * f) / (dx ** 2)
            lap_y = (torch.roll(f, 1, dims=0) + torch.roll(f, -1, dims=0) - 2 * f) / (dy ** 2)
            return lap_x + lap_y

        dudx = grad_x(u); dudy = grad_y(u); adv_u = u * dudx + v * dudy
        dvdx = grad_x(v); dvdy = grad_y(v); adv_v = u * dvdx + v * dvdy

        lap_u = lap(u); lap_v = lap(v)

        u_star = u + dt * (-adv_u + nu * lap_u)
        v_star = v + dt * (-adv_v + nu * lap_v)

        div_star = grad_x(u_star) + grad_y(v_star)

        div_hat = torch.fft.fftn(div_star)
        phi_hat = -div_hat / ksq_torch  # Fix: rhs=∇·u*, negative sign ensures
        phi = torch.fft.ifftn(phi_hat).real

        dphi_dx = grad_x(phi); dphi_dy = grad_y(phi)

        u_new = u_star - dphi_dx  # Fix: no dt
        v_new = v_star - dphi_dy

        velocity = torch.stack([u_new, v_new], dim=0).unsqueeze(0)
        velocity = torch.clamp(velocity, -20, 20)  # Loose clamp
        
        if (step + 1) % (num_steps // 10) == 0:
            print(f"Completed {step + 1}/{num_steps} steps (t={t0 + (step+1)*dt:.4f})", end='\r')

    print("\nN-S simulation complete.")

    velocity_solution = np.stack(vel_traj, axis=0)
    u_solution = velocity_solution[:, 0]
    v_solution = velocity_solution[:, 1]
    
    result = {
        # 'velocity_solution': velocity_solution,
        'x_solution': u_solution,
        'y_solution': v_solution,
        'solution': velocity_solution,
        'x': x, 'y': y, 't_eval': t_eval,
        'parameters': {'nu': nu, 'U_max': U_max, 'Lx': Lx, 'Ly': Ly, 't0': t0, 't_end': t_end, 
                       'dt': dt, 'num_steps': num_steps, 'Nt_sampled': Nt + 1, 'dx': dx, 'dy': dy, 'seed': seed,
                       'Nx': Nx, 'Ny': Ny, 'Nt': Nt,}
    }

    if save_path:
        np.savez(save_path, **result)
        print(f"Saved to {save_path}")

    print(f"Shape (velocity/u/v): {velocity_solution.shape}")
    return result

def generate_2d_convection_diffusion_analytical(
    Lx=1.0, Ly=1.0, t0=0.0, t_end=1.0, 
    Nx=128, Ny=128, Nt=100, 
    nu=0.01, cx=0.5, cy=0.0, sigma0 = 0.05,
    save_path=None
):
    """
    Generates the analytical solution for the 2D convection-diffusion equation (moving Gaussian wave packet).

    The analytical solution is based on the Gaussian solution in an unbounded domain:
    u(x, y, t) = A(t) * exp( - ( (x - x_c(t))^2 + (y - y_c(t))^2 ) / (4 * nu * (t + t_start)) )
    
    Where the center point (x_c, y_c) moves over time with velocity (cx, cy).

    Parameters:
    - nu: Diffusion coefficient (suggest 0.001 - 0.01 to maintain wave shape)
    - cx, cy: Convection velocity (suggest cx=0.5, cy=0 to observe significant lateral movement)
    
    - sigma0: Initial variance (determines wave packet width). Smaller values mean sharper packets. 
      Set small to avoid touching boundaries.
    """
    
    # 1. Generate grid
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    t_eval = np.linspace(t0, t_end, Nt + 1) 
    X, Y = np.meshgrid(x, y) # Shape (Ny, Nx)

    # 2. Define Gaussian packet parameters
    # Initial center position (x0, y0)
    # Placed at a left-leaning position (0.25, 0.5) so it has space to move right
    x0 = 0.25 * Lx
    y0 = 0.5 * Ly
    
    
    # Define analytical solution function
    def gaussian_packet(X, Y, t, cx, cy, nu, x0, y0, sigma0):
        """
        X, Y: Grid
        t: Current time
        cx, cy: Velocity
        nu: Diffusion coefficient
        x0, y0: Initial position
        sigma0: Initial width standard deviation parameter
        """
        # Calculate time-varying center position (Convection effect)
        x_center = x0 + cx * t
        y_center = y0 + cy * t
        
        # Calculate time-varying diffusion width (Diffusion effect)
        # As time t increases, the denominator grows, packet widens, peak drops
        # Formula derived from the fundamental solution of the heat kernel
        sigma_sq_t = sigma0**2 + 2 * nu * t
        
        # Calculate Gaussian distribution
        # For numerical stability, we don't use the 1/(4*pi*nu*t) normalization coefficient, 
        # but keep the peak within a reasonable range. We focus on the evolution of shape.
        exponent = -((X - x_center)**2 + (Y - y_center)**2) / (2 * sigma_sq_t)
        u = np.exp(exponent)
        
        return u

    # 3. Calculate solution for all time steps
    u_analytical = np.zeros((Nt + 1, Ny, Nx))
    
    print(f"Generating data with: nu={nu}, cx={cx}, cy={cy}")
    print(f"Initial Center: ({x0}, {y0}) -> Expected Final Center: ({x0 + cx*t_end}, {y0 + cy*t_end})")

    for i, t in enumerate(t_eval):
        u_analytical[i] = gaussian_packet(X, Y, t, cx, cy, nu, x0, y0, sigma0)

    # 4. Assemble output
    result = {
        'solution': u_analytical,
        'x': x,
        'y': y,
        't_eval': t_eval,
        'parameters': {
            'Lx': Lx, 'Ly': Ly, 
            'Nx': Nx, 'Ny': Ny, 
            'Nt': Nt,
            'nu': nu, 'cx': cx, 'cy': cy
        }
    }

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        np.savez(save_path, **result)
        print(f"Results saved to {save_path}")

    return result

# Generate data and visualize
if __name__ == "__main__":
    # Parameter settings
    Lx, Ly = 1.0, 1.0
    Nx, Ny = 32, 32
    Nt = 100

    # 2D wave equation analytical solution
    wave_data = generate_2d_wave_analytical(
        Lx=Lx, Ly=Ly, 
        t0=0, t_end=2, 
        Nx=Nx, Ny=Ny, 
        Nt=Nt, 
        c=1.0, 
        save_path="data/wave_solution_2d(32).npz"  # Optional save
    )
    wave_u = wave_data['solution']
    wave_t = wave_data['t_eval']
    create_interactive_wave_viz(wave_u, wave_t, Lx, Ly, interval=0.05)

    # # 2D convection-diffusion equation analytical solution
    # wave_data = generate_2d_convection_diffusion_analytical(
    #     Lx=Lx, Ly=Ly, 
    #     t0=0, t_end=2, 
    #     Nx=Nx, Ny=Ny, 
    #     Nt=Nt, 
    #     cy=0.09,cx=0.18,nu=0.002,
    #     sigma0=0.1,
    #     save_path="data/convection_diffusion_2d(64).npz"  # Optional save
    # )
    # wave_u = wave_data['solution']
    # wave_t = wave_data['t_eval']
    # create_interactive_wave_viz(wave_u, wave_t, Lx, Ly, interval=0.05)

    # # Generate 2D Navier-Stokes Taylor-Green vortex
    # ns_data = generate_2d_naviers_stokes(
    #     Lx=2 * np.pi, Ly=2 * np.pi, 
    #     t0=0, t_end=10.0, 
    #     Nx=Nx, Ny=Ny, Nt=Nt,
    #     nu=0.05, U_max=1.0, seed=42,
    #     save_path="data/naviers_stokes_2d(32).npz"
    #     )
    # create_interactive_wave_viz(ns_data['x_solution'], ns_data['t_eval'], 2*np.pi, 2*np.pi, interval=0.1)