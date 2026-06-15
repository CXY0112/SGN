####################################
#       用于生成训练的数据          #
####################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from torch import nn  # 用于 Conv2d 拉普拉斯
import torch  # 新增：GPU 加速
import os
import time

# 可视化2维数据
def create_interactive_wave_viz(u_analytical, t_eval, Lx=1.0, Ly=1.0, interval=0.05,vmin=None, vmax=None):
    """
    创建交互式2D波方程可视化API。
    
    参数:
    - u_analytical: np.ndarray, 形状 (Nt, Ny, Nx)，时空波场数据。
    - t_eval: np.ndarray, 形状 (Nt,)，时间数组。
    - Lx, Ly: float, 空间域大小，默认1.0。
    - interval: float, 播放间隔（秒），默认0.05（约50ms/帧）。
    
    返回: 无，直接显示交互窗口。
    """
    Nt, Ny, Nx = u_analytical.shape
    plt.ion()  # 启用交互模式
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)  # 为控件留空间

    # 自动计算色域范围（如未指定）
    if vmin is None:
        vmin = float(np.min(u_analytical))
    if vmax is None:
        vmax = float(np.max(u_analytical))

    # 播放控制变量
    playing = False

    # 初始绘制函数
    def init_plot(frame_idx):
        ax.clear()
        im = ax.imshow(u_analytical[frame_idx], extent=[0, Lx, 0, Ly], origin='lower',
                       cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f't = {t_eval[frame_idx]:.2f}')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        # 添加颜色条（只一次）
        if not hasattr(init_plot, 'cbar'):
            init_plot.cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        return im

    # 初始帧
    init_frame = 0
    im = init_plot(init_frame)

    # 添加 Slider
    ax_slider = plt.axes([0.15, 0.12, 0.65, 0.03])
    slider = Slider(ax_slider, 'Frame (t)', 0, Nt - 1, valinit=init_frame, valstep=1)

    # 更新函数
    def update(val):
        frame_idx = int(slider.val)
        init_plot(frame_idx)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # 播放函数
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

    # 暂停函数
    def pause(event):
        nonlocal playing
        playing = False

    # 添加按钮
    ax_play = plt.axes([0.15, 0.05, 0.1, 0.04])
    play_button = Button(ax_play, 'Play')
    play_button.on_clicked(play)

    ax_pause = plt.axes([0.35, 0.05, 0.1, 0.04])
    pause_button = Button(ax_pause, 'Pause')
    pause_button.on_clicked(pause)

    plt.show(block=True)  # 阻塞显示，确保窗口保持打开

def generate_2d_wave_analytical(Lx=1.0, Ly=1.0, t0=0.0, t_end=2.0, Nx=128, Ny=128, Nt=100, c=1.0, save_path=None):
    """
    生成二维波动方程基模解析解的数值数据。

    该函数计算二维波动方程在矩形域 [0, Lx] × [0, Ly] 上，带有 Dirichlet 边界条件 (u=0 on boundaries) 的
    基模 (fundamental mode) 解析解。该解通过分离变量法得到，是一个驻波解，用于描述波在域内的振荡。

    具体解析解公式：

    $$ u(x, y, t) = \sin\left( \frac{\pi x}{L_x} \right) \sin\left( \frac{\pi y}{L_y} \right) \cos\left( \frac{\pi c t}{L_x} \right) $$

    其中：
    - 空间部分确保边界为零，形成驻波模式。
    - 时间部分为简谐振荡，频率由 $c$ 和 $L_x$ 决定（周期 $T = 2 L_x / c$）。

    参数:
    - Lx, Ly: float, 空间域长度 [0, Lx] x [0, Ly]（默认1.0）。
    - t0, t_end: float, 时间区间起点和终点（默认0, 2）。
    - Nx, Ny: int, 空间离散点数（默认128）。
    - Nt: int, 时间离散点数（默认100）。
    - c: float, 波速（默认1.0）。
    - save_path: str, 可选NPZ保存路径（默认None，不保存）。

    返回:
    - dict: {'solution': u_analytical (Nt, Ny, Nx), 'x': x_array, 'y': y_array, 't_eval': t_array}
    """
    # 生成网格
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    t_eval = np.linspace(t0, t_end, Nt+1)
    X, Y = np.meshgrid(x, y)  # 2D 空间网格

    # 定义解析解：分离变量基模
    def analytical_solution(X, Y, t, c, Lx, Ly):
        return np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly) * np.cos(np.pi * c * t / Lx)

    # 计算解析解（形状: (Nt, Ny, Nx)）
    u_analytical = np.zeros((Nt+1, Ny, Nx))
    for i, t in enumerate(t_eval):
        u_analytical[i] = analytical_solution(X, Y, t, c, Lx, Ly)

    # 组装输出字典
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

    # 可选：保存到NPZ文件
    if save_path is not None:
        np.savez(save_path, **result)
        print(f"结果已保存至 {save_path}")

    print(f"Solution shape: {u_analytical.shape}")
    return result

def generate_2d_fisher_kpp_analytical(Lx=1.0, Ly=1.0, t0=0.0, t_end=2.0, Nx=128, Ny=128, Nt=100, D=0.1, r=1.0, save_path=None):
    """
    生成二维Fisher-KPP方程线性化基模解析解的数值数据。

    该函数计算二维Fisher-KPP方程在矩形域 [0, Lx] × [0, Ly] 上，带有 Dirichlet 边界条件 (u=0 on boundaries) 的
    线性化基模 (fundamental mode) 解析解。该解通过分离变量法对小扰动 (u << 1) 线性化得到，用于描述种群/反应前沿的不稳定性演化。

    具体解析解公式（线性化近似）：

    $$ u(x, y, t) = \sin\left( \frac{\pi x}{L_x} \right) \sin\left( \frac{\pi y}{L_y} \right) \exp\left[ \left( -D \pi^2 \left( \frac{1}{L_x^2} + \frac{1}{L_y^2} \right) + r \right) t \right] $$

    其中：
    - 空间部分确保边界为零，形成驻波模式。
    - 时间部分为指数增长/衰减，增长率 λ = r - D π² (1/Lx² + 1/Ly²)（若 λ > 0，则不稳定增长）。
    - 注意：此为线性近似；非线性饱和 (1-u 项) 需数值方法。

    参数:
    - Lx, Ly: float, 空间域长度 [0, Lx] x [0, Ly]（默认1.0）。
    - t0, t_end: float, 时间区间起点和终点（默认0, 2）。
    - Nx, Ny: int, 空间离散点数（默认128）。
    - Nt: int, 时间离散点数（默认100）。
    - D: float, 扩散系数（默认0.1）。
    - r: float, 反应增长率（默认1.0）。
    - save_path: str, 可选NPZ保存路径（默认None，不保存）。

    返回:
    - dict: {'solution': u_analytical (Nt, Ny, Nx), 'x': x_array, 'y': y_array, 't_eval': t_array}
    """
    # 生成网格
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    t_eval = np.linspace(t0, t_end, Nt+1)
    X, Y = np.meshgrid(x, y)  # 2D 空间网格

    # 定义解析解：线性化基模
    def analytical_solution(X, Y, t, D, r, Lx, Ly):
        lambda_mn = np.pi**2 * (1/Lx**2 + 1/Ly**2)
        growth_rate = -D * lambda_mn + r
        return np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly) * np.exp(growth_rate * t)

    # 计算解析解（形状: (Nt, Ny, Nx)）
    u_analytical = np.zeros((Nt+1, Ny, Nx))
    for i, t in enumerate(t_eval):
        u_analytical[i] = analytical_solution(X, Y, t, D, r, Lx, Ly)

    # 组装输出字典
    result = {
        'solution': u_analytical,
        'x': x,
        'y': y,
        't_eval': t_eval,
        'parameters':{
            'Lx': Lx,
            'Ly': Ly,
            't0': t0,
            't_end': t_end,
            'Nx': Nx,
            'Ny': Ny,
            'Nt': Nt,
            'D': D,
            'r': r
        }
    }

    # 可选：保存到NPZ文件
    if save_path is not None:
        np.savez(save_path, **result)
        print(f"结果已保存至 {save_path}")

    print(f"Solution shape: {u_analytical.shape}")
    print(f"增长率 λ: {-D * np.pi**2 * (1/Lx**2 + 1/Ly**2) + r:.4f} (若 >0，则指数增长)")
    return result

def solve_2d_fisher_kpp_numerical(Lx=10.0, Ly=10.0, t0=0.0, t_end=2.0, Nx=128, Ny=128, Nt=100, D=0.1, r=1.0, u_initial=None, save_path=None):
    """
    使用显式有限差分法 (FDM) 求解二维 Fisher-KPP 方程的数值解。

    该函数通过时间上的显式欧拉法和空间上的二阶中心差分，模拟二维空间内
    种群增长或化学反应前沿的扩散和非线性饱和过程。由于数值求解的确定性，
    且初始条件默认设置为确定性的高斯扰动，生成的实验数据具有高可复现性。

    具体方程（Fisher-KPP 方程）：
    $$ \frac{\partial u}{\partial t} = D \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) + r u (1-u) $$

    边界条件和初始条件：
    - 边界条件：Dirichlet 边界 (u=0) 施加在矩形域 [0, Lx] x [0, Ly] 的所有边界上。
    - 初始条件：默认使用域中心的一个确定的、微小的高斯扰动。
    
    注意：此显式方法受 CFL 稳定性条件严格限制，需要 dt ≤ 0.5 * min(dx², dy²) / D。
    若 dt 过大，数值解将发散。

    参数:
    - Lx, Ly: float, 空间域在 x 和 y 方向的长度（默认10.0）。
    - t0: float, 模拟起始时间（默认0.0）。
    - t_end: float, 模拟结束时间（默认2.0）。
    - Nx, Ny: int, 空间 x 和 y 方向的离散点数（默认128）。
    - Nt: int, 时间步数。总时间步长 dt = (t_end - t0) / Nt（默认100）。
    - D: float, 扩散系数（默认0.1）。
    - r: float, 反应项的线性增长率（默认1.0）。
    - u_initial: ndarray (Ny, Nx), 可选。如果提供，则使用此数组作为初始条件（默认None）。
    - save_path: str, 可选。NPZ 格式数据的保存路径（默认None，不保存）。

    返回:
    - dict: 包含求解结果和网格信息的字典。
        - 'solution': u_history (Nt+1, Ny, Nx)，包含从 t0 到 t_end 所有时间步的解。
        - 'x': x_array (Nx)，x 坐标轴上的离散点。
        - 'y': y_array (Ny)，y 坐标轴上的离散点。
        - 't_eval': t_array (Nt+1)，模拟过程中的所有时间点。
        - 'parameters': dict, 包含所有关键参数 (D, r, Lx, Ly, dt, dx, dy 等)，用于数据可复现性。
    """
    # 1. 离散化参数
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    # 2. 使用正确的时间跨度计算 dt
    dt = (t_end - t0) / Nt 
    
    # FTCS 稳定性检查
    cfl_limit = 0.5 * min(dx**2, dy**2) / D
    if dt > cfl_limit:
        print(f"⚠️ 警告: 显式方法不稳定！需要 dt < {cfl_limit:.6f}，当前 dt={dt:.6f}。")
        print("请增加 Nt 或减小 D。")
    
    # 2. 网格初始化
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    # 1. 时间点数与解的维度 (Nt+1) 匹配
    t_eval = np.linspace(t0, t_end, Nt+1) 
    
    # 3. 初始条件 (逻辑保持不变，确保了确定性)
    if u_initial is None:
        u = np.zeros((Ny, Nx))
        X, Y = np.meshgrid(x, y)
        u_initial_pert = 0.01 * np.exp(-100 * ((X - Lx/2)**2 + (Y - Ly/2)**2))
        u[:] = u_initial_pert
    else:
        if u_initial.shape != (Ny, Nx):
            raise ValueError(f"初始条件形状不匹配。期望 ({Ny}, {Nx})，得到 {u_initial.shape}")
        u = u_initial.copy()

    # 确保初始边界为零
    u[0, :], u[-1, :], u[:, 0], u[:, -1] = 0.0, 0.0, 0.0, 0.0

    # 存储所有时间步的解
    u_history = np.zeros((Nt + 1, Ny, Nx))
    u_history[0] = u.copy()
    
    print(f"开始 FDM 求解: Lx={Lx}, Ly={Ly}, Nx={Nx}, Ny={Ny}, Nt={Nt}, t0={t0}, t_end={t_end}, dt={dt:.6f}")

    # 4. 时间步进 (显式欧拉法)
    for k in range(Nt):
        u_new = u.copy()
        
        # 空间离散化 (中心差分, 仅计算内部点)
        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                # 扩散项 (Laplacian)
                laplacian = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dx**2 + \
                            (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dy**2
                
                # 反应项 (Non-linear KPP term)
                reaction = r * u[i, j] * (1.0 - u[i, j])
                
                # 显式欧拉时间步进
                u_new[i, j] = u[i, j] + dt * (D * laplacian + reaction)

        # 更新解
        u = u_new
        
        # 边界条件 (保持 u=0),防御性编程
        u[0, :], u[-1, :], u[:, 0], u[:, -1] = 0.0, 0.0, 0.0, 0.0
        
        # 存储结果
        u_history[k+1] = u.copy()
        
        # if (k + 1) % (Nt // 10 if Nt >= 10 else 1) == 0:
        #     # 打印当前时间点的索引 k+1，对应 t_eval[k+1]
        #     print(f"完成 {k+1}/{Nt} 步 (t={t_eval[k+1]:.4f})", end='\r')

    print("\nFDM 求解完成。")

    # 5. 组装输出字典 (包含 t0)
    result = {
        'solution': u_history,
        'x': x,
        'y': y,
        't_eval': t_eval,
        'parameters': {'D': D, 'r': r, 'Lx': Lx, 'Ly': Ly, 't0': t0, 't_end': t_end, 'dt': dt, 'dx': dx, 'dy': dy, 'Nt': Nt, 'Nx': Nx, 'Ny': Ny}
    }

    # 可选：保存到NPZ文件
    if save_path is not None:
        np.savez(save_path, **result)
        print(f"结果已保存至 {save_path}")

    print(f"Solution shape: {u_history.shape}")
    return result

def generate_2d_turing_analytical(Lx=10.0, Ly=10.0, t0=0.0, t_end=50.0, Nx=128, Ny=128, Nt=100, 
                                  D_u=0.1, D_v=1.0, a=0.1, b=0.9, save_path=None, noise_amp=0.01, seed=42):
    """
    使用显式有限差分法（基于 PyTorch/CUDA 加速）求解二维 Schnakenberg（Turing）反应-扩散系统，
    生成自组织斑点图案的数值数据。
    
    该模型采用标准的 Schnakenberg 形式，其中反应项没有显式的全局乘数 r。

    数学模型（Schnakenberg 反应，无全局 r）：
    $$ \partial_t u = D_u \nabla^2 u + (a - u + u^2 v) $$
    $$ \partial_t v = D_v \nabla^2 v + (b - u^2 v) $$

    数值方法：显式欧拉时间步进，空间上使用周期性边界条件下的二阶中心差分。

    参数:
    - Lx, Ly: float, 空间域在 x 和 y 方向的长度（默认10.0）。
    - t0, t_end: float, 模拟起始和结束时间。默认 t_end 设为 50.0 以充分演化。
    - Nx, Ny: int, 空间离散点数（默认256）。
    - Nt: int, 采样的输出时间点数。总步数 num_steps = Nt * 积分倍率（默认500）。
    - D_u, D_v: float, 扩散系数（v > u 是图灵不稳定的必要条件）。
    - a, b: float, Schnakenberg 模型参数。
    - save_path: str, 可选。NPZ 保存路径。
    - noise_amp: float, 初始噪声幅度，用于触发不稳定（默认0.01）。
    - seed: int, 随机种子（默认42，保证可复现）。

    返回:
    - dict: 包含求解结果和网格信息的字典，维度 (Nt+1, Ny, Nx)。
    """
    # 1. 初始化和可复现性设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 固定种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    Nt = Nt + 1
    # 积分步数和时间步长
    integration_factor = 2000 # 使用 20 倍的细化时间步长进行积分
    num_steps = Nt * integration_factor
    dt = (t_end - t0) / num_steps

    # 2. 网格和稳定性检查
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    dx = Lx / (Nx - 1) 
    dy = Ly / (Ny - 1)
    
    t_eval = np.linspace(t0, t_end, Nt + 1) 
    t_eval = t_eval[1:]
    # CFL 稳定性检查（针对扩散项）
    D_max = max(D_u, D_v)
    cfl_limit = 0.25 * min(dx**2, dy**2) / D_max
    if dt > cfl_limit:
        print(f"⚠️ 警告: 显式方法不稳定！CFL 建议 dt < {cfl_limit:.6f}，当前 dt={dt:.6f}。")
        print("请增加 Nt 或减小 D_u/D_v。")
    print(f"积分步长 dt: {dt:.6f}，总积分步数: {num_steps}")

    # 3. 初始条件（稳态 + 噪声）
    # 正确的 Schnakenberg 稳态：u_eq = a + b, v_eq = b / u_eq^2
    u_eq = a + b
    v_eq = b / (u_eq**2)

    u0_np = np.full((Ny, Nx), u_eq) + noise_amp * np.random.randn(Ny, Nx)
    v0_np = np.full((Ny, Nx), v_eq) + noise_amp * np.random.randn(Ny, Nx)
    
    # 转换到 PyTorch Tensor (维度: (1, 1, Ny, Nx))
    u = torch.tensor(u0_np, dtype=torch.float32, device=device)[None, None, :, :] 
    v = torch.tensor(v0_np, dtype=torch.float32, device=device)[None, None, :, :]

    # 4. 轨迹存储与采样索引
    traj = []  # 改为一次性存 (2, Ny, Nx)
    
    sample_indices = np.round(np.linspace(0, num_steps, Nt + 1, endpoint=True)).astype(int)
    sample_index_set = set(sample_indices)

    # 5. Euler 循环 (时间步进)
    for step in range(num_steps + 1):
        
        # 采样点存储 (包含 t0)
        if step in sample_index_set:
            u_np = u.squeeze(0).squeeze(0).cpu().numpy()
            v_np = v.squeeze(0).squeeze(0).cpu().numpy()
            traj.append(np.stack([u_np, v_np], axis=0))  # (2, Ny, Nx)
        
        if step == num_steps: # 最后一个步长不进行更新，只存储
            break

        # 手动计算拉普拉斯 (周期性边界条件)
        lap_u = (torch.roll(u, 1, dims=2) + torch.roll(u, -1, dims=2) + 
                 torch.roll(u, 1, dims=3) + torch.roll(u, -1, dims=3) - 4 * u) / (dx**2)
        lap_v = (torch.roll(v, 1, dims=2) + torch.roll(v, -1, dims=2) + 
                 torch.roll(v, 1, dims=3) + torch.roll(v, -1, dims=3) - 4 * v) / (dx**2)

        # 反应项 【核心修改：移除 r】
        u_sq = u.squeeze()
        v_sq = v.squeeze()
        f = (a - u_sq + u_sq**2 * v_sq)  # 反应项 f(u, v) for u (无 r)
        g = (b - u_sq**2 * v_sq)         # 反应项 g(u, v) for v (无 r)

        # 显式欧拉步 (du/dt = D * Lap + Reaction)
        du = dt * (D_u * lap_u.squeeze() + f)
        dv = dt * (D_v * lap_v.squeeze() + g)
        
        u = u + du.unsqueeze(0).unsqueeze(0)
        v = v + dv.unsqueeze(0).unsqueeze(0)
        
        # 物理 Clamping：仅防止浓度为负
        u = u.clamp(min=0)
        v = v.clamp(min=0)
        
        if (step + 1) % (num_steps // 10 if num_steps >= 10 else 1) == 0:
            print(f"完成 {step + 1}/{num_steps} 步 (t={t0 + (step+1)*dt:.4f})", end='\r')

    print("\nTuring 模拟求解完成。")

    # 6. 数据整理和输出
    solution = np.stack(traj, axis=0)       # 新增：(Nt+1, 2, Ny, Nx)
    solution= solution[1:,:,:,:]
    u_solution = solution[:, 0, :, :]       # (Nt+1, Ny, Nx)
    v_solution = solution[:, 1, :, :]       # (Nt+1, Ny, Nx)
    
    # 组装输出字典
    result = {
        'x_solution': u_solution,
        'y_solution': v_solution,
        'solution': solution,      # 现在是 (Nt+1, 2, Ny, Nx)，与 NS 生成器完全一致
        'x': x,
        'y': y,
        't_eval': t_eval,
        'parameters': {
            'D_u': D_u, 'D_v': D_v, 'a': a, 'b': b, # 【关键：r 已移除】
            'Lx': Lx, 'Ly': Ly, 't0': t0, 't_end': t_end, 
            'dt_total': dt, 'num_steps': num_steps, 'Nt_sampled': Nt,
            'dx': dx, 'dy': dy, 'noise_amp': noise_amp, 'seed': seed,
            'Nx':Nx, 'Ny':Ny
        }
    }

    if save_path is not None:
        np.savez(save_path, **result)
        print(f"结果已保存至 {save_path}")

    print(f"Solution shape (u/v/solution): {u_solution.shape} / {solution.shape}")
    return result

def generate_2d_gray_scott(Lx=10.0, Ly=10.0, t0=0.0, t_end=20.0, Nx=128, Ny=128, Nt=100, 
                           D_U=0.16, D_V=0.08, F=0.035, k=0.06, noise_amp=0.02, save_path=None, seed=42):
    """
    使用显式有限差分法（基于 PyTorch/CUDA 加速）求解二维 Gray-Scott 反应-扩散系统，
    生成自组织图案的数值数据。

    该函数模拟从 U≈1, V≈0（添加随机噪声）开始的演化过程，捕捉非线性效应，
    形成复杂的斑点/条纹图案。代码通过固定随机种子确保了数据的高可复现性。

    数学模型（Gray-Scott 反应）：
    $$ \partial_t U = D_U \nabla^2 U - U V^2 + F (1 - U) $$
    $$ \partial_t V = D_V \nabla^2 V + U V^2 - (F + k) V $$

    数值方法：显式欧拉时间步进，空间上使用周期性边界条件下的二阶中心差分。

    参数:
    - Lx, Ly: float, 空间域长度 [0, Lx] x [0, Ly]（默认10.0）。
    - t0, t_end: float, 模拟起始和结束时间（默认0, 20）。
    - Nx, Ny: int, 空间离散点数（默认256）。
    - Nt: int, 采样的输出时间点数。总步数 num_steps = Nt * 积分倍率（默认100）。
    - D_U, D_V: float, 扩散系数（默认0.16, 0.08）。
    - F: float, 进料率（默认0.035）。
    - k: float, 杀灭率（默认0.06）。
    - noise_amp: float, 初始噪声幅度（默认0.02）。
    - save_path: str, 可选NPZ保存路径（默认None）。
    - seed: int, 随机种子（默认42，保证可复现）。

    返回:
    - dict: 包含求解结果和网格信息的字典，维度 (Nt+1, Ny, Nx)。
    """
    # 1. 初始化和可复现性设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 固定种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 积分步数和时间步长
    integration_factor = 20*750 # 默认使用 20*750 倍的细化时间步长进行积分
    num_steps = Nt * integration_factor
    dt = (t_end - t0) / num_steps

    # 2. 网格和稳定性检查
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    dx = Lx / (Nx - 1) 
    dy = Ly / (Ny - 1)
    
    # t_eval 维度匹配 Nt+1
    t_eval = np.linspace(t0, t_end, Nt + 1) 

    # CFL 稳定性检查（针对扩散项）
    D_max = max(D_U, D_V)
    # 稳定条件大致是 dt <= 0.25 * min(dx^2, dy^2) / D_max
    cfl_limit = 0.25 * min(dx**2, dy**2) / D_max
    if dt > cfl_limit:
        print(f"⚠️ 警告: 显式方法不稳定！CFL 建议 dt < {cfl_limit:.6f}，当前 dt={dt:.6f}。")
        print("请增加 Nt 或减小 D_U/D_V。")
    print(f"积分步长 dt: {dt:.6f}，总积分步数: {num_steps}")

    # 3. 初始条件：U = 1 - noise, V = noise
    noise_np = noise_amp * np.random.randn(Ny, Nx)
    u0_np = 1.0 - noise_np
    v0_np = noise_np
    
    # 转换到 PyTorch Tensor (维度: (1, 1, Ny, Nx))
    u = torch.tensor(u0_np, dtype=torch.float32, device=device)[None, None, :, :] 
    v = torch.tensor(v0_np, dtype=torch.float32, device=device)[None, None, :, :]

    # 4. 轨迹存储与采样索引
    u_traj = []
    v_traj = []
    
    # 【修正 2】采样索引：确保包含 t0 (k=0) 和 t_end，总长 Nt+1
    sample_indices = np.round(np.linspace(0, num_steps, Nt + 1, endpoint=True)).astype(int)
    sample_index_set = set(sample_indices)

    # 5. Euler 循环 (时间步进)
    for step in range(num_steps + 1):
        
        # 采样点存储 (包含 t0)
        if step in sample_index_set:
            u_traj.append(u.squeeze().cpu().numpy())
            v_traj.append(v.squeeze().cpu().numpy())
        
        if step == num_steps: # 最后一个步长不进行更新，只存储
            break

        # 手动计算拉普拉斯 (周期性边界条件)
        lap_u = (torch.roll(u, 1, dims=2) + torch.roll(u, -1, dims=2) + 
                 torch.roll(u, 1, dims=3) + torch.roll(u, -1, dims=3) - 4 * u) / (dx**2)
        lap_v = (torch.roll(v, 1, dims=2) + torch.roll(v, -1, dims=2) + 
                 torch.roll(v, 1, dims=3) + torch.roll(v, -1, dims=3) - 4 * v) / (dy**2)

        # 反应项 (Gray-Scott)
        u_sq = u.squeeze()
        v_sq = v.squeeze()
        uv2 = u_sq * v_sq**2
        f = -uv2 + F * (1.0 - u_sq)  # 反应项 f(U, V) for U
        g = uv2 - (F + k) * v_sq     # 反应项 g(U, V) for V

        # 显式欧拉步 (du/dt = D * Lap + Reaction)
        du = dt * (D_U * lap_u.squeeze() + f)
        dv = dt * (D_V * lap_v.squeeze() + g)
        
        u = u + du.unsqueeze(0).unsqueeze(0)
        v = v + dv.unsqueeze(0).unsqueeze(0)
        
        # 物理 Clamping：防止负值，U的上限通常为1
        # u = u.clamp(0, 1) # U 通常不超过 1 
        # v = v.clamp(min=0) # V 浓度必须非负
        
        if (step + 1) % (num_steps // 10 if num_steps >= 10 else 1) == 0:
            print(f"完成 {step + 1}/{num_steps} 步 (t={t0 + (step+1)*dt:.4f})", end='\r')

    print("\nGray-Scott 模拟求解完成。")

    # 6. 数据整理和输出
    u_solution = np.stack(u_traj, axis=0)
    v_solution = np.stack(v_traj, axis=0)
    
    # 组装输出字典
    result = {
        'u_solution': u_solution,
        'v_solution': v_solution,
        'solution': u_solution, 
        'x': x,
        'y': y,
        't_eval': t_eval,
        'parameters': {
            'D_U': D_U, 'D_V': D_V, 'F': F, 'k': k, 
            'Lx': Lx, 'Ly': Ly, 't0': t0, 't_end': t_end, 
            'dt_total': dt, 'num_steps': num_steps, 'Nt_sampled': Nt + 1,
            'dx': dx, 'dy': dy, 'noise_amp': noise_amp, 'seed': seed
        }
    }

    if save_path is not None:
        np.savez(save_path, **result)
        print(f"结果已保存至 {save_path}")

    print(f"Solution shape (U/V/solution): {u_solution.shape}")
    return result

def generate_2d_naviers_stokes(Lx=2 * np.pi, Ly=2 * np.pi, t0=0.0, t_end=10.0, Nx=128, Ny=128, Nt=100, 
                               nu=0.01, U_max=1.0, save_path=None, seed=42):
    """
    使用投影方法（基于 PyTorch/CUDA 加速）求解二维不可压缩纳维-斯托克斯方程，
    以 Taylor-Green 涡旋作为初始条件，生成周期性边界下的衰减涡旋流动数据。

    数学模型（2D 不可压缩 N-S）：
    $$ \nabla \cdot \mathbf{u} = 0 $$
    $$ \partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u} $$

    数值方法：Chorin 投影法（低阶变体），显式欧拉时间步，二阶中心差分 + FFT Poisson。
    边界：周期性。修复版确保能量守恒/衰减物理合理。

    参数: (同前)
    返回: dict with velocity_solution (Nt+1, 2, Ny, Nx) 等。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    integration_factor = 200  # 细化步长，提高稳定性
    num_steps = Nt * integration_factor
    dt = (t_end - t0) / num_steps

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    dx = Lx / Nx
    dy = Ly / Ny
    
    t_eval = np.linspace(t0, t_end, Nt + 1) 

    # 增强 CFL（保守系数 0.4）
    cfl_adv = 0.4 * min(dx, dy) / U_max
    cfl_diff = 0.4 * min(dx**2, dy**2) / nu
    cfl_limit = min(cfl_adv, cfl_diff)
    if dt > cfl_limit:
        print(f"⚠️ 警告: CFL 建议 dt < {cfl_limit:.6f}，当前 {dt:.6f}。建议增大 integration_factor。")
    print(f"dt: {dt:.6f}，总步: {num_steps}")

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
        phi_hat = -div_hat / ksq_torch  # 修复：rhs=∇·u*，负号确保
        phi = torch.fft.ifftn(phi_hat).real

        dphi_dx = grad_x(phi); dphi_dy = grad_y(phi)

        u_new = u_star - dphi_dx  # 修复：无 dt
        v_new = v_star - dphi_dy

        velocity = torch.stack([u_new, v_new], dim=0).unsqueeze(0)
        velocity = torch.clamp(velocity, -20, 20)  # 宽松 clamp
        
        if (step + 1) % (num_steps // 10) == 0:
            print(f"完成 {step + 1}/{num_steps} 步 (t={t0 + (step+1)*dt:.4f})", end='\r')

    print("\nN-S 模拟求解完成。")

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
        print(f"保存至 {save_path}")

    print(f"形状 (velocity/u/v): {velocity_solution.shape}")
    return result

def generate_2d_burgers_data_numerical(Lx=2 * np.pi, Ly=2 * np.pi, t0=0.0, t_end=4.0, 
                                       Nx=32, Ny=32, Nt=400, nu=0.01, 
                                       noise_level=0.001, precision=1,seed=None, # 将seed设为None，每次调用可生成不同数据
                                       save_path=None):
    """
    生成一个二维 Burgers' 方程的独立数据集样本，模拟 PDE-Net 2.0 论文中的数据生成过程。

    通过高精度数值求解（高分辨率网格和细时间步），然后降采样和加噪，得到最终数据集。
    该函数只生成**一个样本**。如需多个样本，请在外部多次调用。

    方程:
    ∂t u = nu * ∇²u - (u ∂x u + v ∂y u)
    ∂t v = nu * ∇²v - (u ∂x v + v ∂y v)
    系统采用周期性边界条件。

    参数:
    - Lx, Ly: float, 空间域长度 [0, Lx] x [0, Ly] (默认 2*pi)。
    - t0, t_end: float, 时间区间起点和终点（默认0, 4.0）。
    - Nx, Ny: int, 最终**输出**空间网格点数 (论文中为 32)。
    - Nt: int, 最终**输出**的时间采样点数 (论文中为 400)。
    - nu: float, 粘性系数 (扩散项系数，默认0.1)。
    - noise_level: float, 噪声水平 (论文中为 0.001)。
    - seed: int or None, 随机种子。如果为 None，则使用当前系统时间作为种子，保证数据随机性。
    - save_path: str, 可选的NPZ保存路径。

    返回:
    - dict: {'solution': (u, v) 组合数据 (Nt+1, Ny, Nx, 2), 'x_solution': u 变量, 'y_solution': v 变量, 
             'x': x_grid, 'y': y_grid, 't_eval': t_grid}
    """
    # 1. 初始化和设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 设置种子：如果 seed 为 None，则使用随机种子以确保每次调用生成不同的初始条件
    if seed is None:
        seed = int(os.getpid() * time.time() * 1000) % 10000
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # --- 2. 内部高精度网格定义 ---
    # 论文：输出网格 32x32, 采样 400 帧。高精度 128x128, 6400 步
    RES_FACTOR = 4 * precision
    TIME_FACTOR = 16 * precision

    Nx_true = Nx * RES_FACTOR
    Ny_true = Ny * RES_FACTOR
    Nt_true = Nt * TIME_FACTOR

    # --- 3. 网格和时间步长 ---
    # 高精度网格
    x_true = np.linspace(0, Lx, Nx_true, endpoint=False)
    y_true = np.linspace(0, Ly, Ny_true, endpoint=False)
    dx = Lx / Nx_true 
    dy = Ly / Ny_true
    dt_true = (t_end - t0) / Nt_true
    
    # 训练网格 (输出网格)
    x_train = np.linspace(0, Lx, Nx, endpoint=False)
    y_train = np.linspace(0, Ly, Ny, endpoint=False)
    t_train = np.linspace(t0, t_end, Nt + 1)
    
    # 降采样索引
    idx_x = np.arange(0, Nx_true, RES_FACTOR)
    idx_y = np.arange(0, Ny_true, RES_FACTOR)
    
    # 稳定性检查 (仅为警告)
    D_max = nu
    cfl_limit_diff = 0.25 * min(dx**2, dy**2) / D_max
    if dt_true > cfl_limit_diff:
         print(f"⚠️ 警告: 扩散项显式稳定性 CFL 建议 dt < {cfl_limit_diff:.6f}，当前 dt={dt_true:.6f}。")
    
    # --- 4. 初始条件生成 ---
    k_max, l_max = 4, 4
    lambda_u = np.random.randn(2 * k_max + 1, 2 * l_max + 1)
    gamma_u = np.random.randn(2 * k_max + 1, 2 * l_max + 1)
    lambda_v = np.random.randn(2 * k_max + 1, 2 * l_max + 1)
    gamma_v = np.random.randn(2 * k_max + 1, 2 * l_max + 1)
    
    c_u = np.random.uniform(-2, 2) 
    c_v = np.random.uniform(-2, 2)
    
    def generate_w0(Lx, Ly, Nx, Ny, lambda_kl, gamma_kl):
        x = np.linspace(0, Lx, Nx, endpoint=False)
        y = np.linspace(0, Ly, Ny, endpoint=False)
        X, Y = np.meshgrid(x, y)
        w0 = np.zeros((Ny, Nx))
        
        for k in range(-k_max, k_max + 1):
            for l in range(-l_max, l_max + 1):
                if k == 0 and l == 0: continue
                idx_k = k + k_max
                idx_l = l + l_max
                w0 += lambda_kl[idx_k, idx_l] * np.cos(k * X + l * Y) + \
                      gamma_kl[idx_k, idx_l] * np.sin(k * X + l * Y)
        return w0

    u0_w0 = generate_w0(Lx, Ly, Nx_true, Ny_true, lambda_u, gamma_u)
    v0_w0 = generate_w0(Lx, Ly, Nx_true, Ny_true, lambda_v, gamma_v)
    
    max_u0_w0 = np.max(np.abs(u0_w0))
    max_v0_w0 = np.max(np.abs(v0_w0))
    
    u_init = (2 * u0_w0 / max_u0_w0) + c_u if max_u0_w0 != 0 else np.full_like(u0_w0, c_u)
    v_init = (2 * v0_w0 / max_v0_w0) + c_v if max_v0_w0 != 0 else np.full_like(v0_w0, c_v)

    # --- 5. 核心求解逻辑 (单样本) ---
    u = torch.tensor(u_init, dtype=torch.float32, device=device)[None, None, :, :]
    v = torch.tensor(v_init, dtype=torch.float32, device=device)[None, None, :, :]
    
    # 存储降采样后的真值 (Nt+1, Ny_true, Nx_true)
    u_series_true = torch.zeros((Nt + 1, Ny_true, Nx_true), device=device)
    v_series_true = torch.zeros((Nt + 1, Ny_true, Nx_true), device=device)

    u_series_true[0] = u.squeeze()
    v_series_true[0] = v.squeeze()
    
    train_steps = np.round(np.linspace(0, Nt_true, Nt + 1)).astype(int)
    
    for t_step in range(1, Nt_true + 1):
        
        # 空间导数 (二阶中心差分, 周期性边界)
        lap_u = (torch.roll(u, 1, dims=2) + torch.roll(u, -1, dims=2) - 2 * u) / (dy**2) + \
                (torch.roll(u, 1, dims=3) + torch.roll(u, -1, dims=3) - 2 * u) / (dx**2)
        lap_v = (torch.roll(v, 1, dims=2) + torch.roll(v, -1, dims=2) - 2 * v) / (dy**2) + \
                (torch.roll(v, 1, dims=3) + torch.roll(v, -1, dims=3) - 2 * v) / (dx**2)

        u_x = (torch.roll(u, -1, dims=3) - torch.roll(u, 1, dims=3)) / (2 * dx)
        u_y = (torch.roll(u, -1, dims=2) - torch.roll(u, 1, dims=2)) / (2 * dy)
        v_x = (torch.roll(v, -1, dims=3) - torch.roll(v, 1, dims=3)) / (2 * dx)
        v_y = (torch.roll(v, -1, dims=2) - torch.roll(v, 1, dims=2)) / (2 * dy)
        
        u_sq = u.squeeze()
        v_sq = v.squeeze()
        
        RHS_u = nu * lap_u.squeeze() - (u_sq * u_x.squeeze() + v_sq * u_y.squeeze())
        RHS_v = nu * lap_v.squeeze() - (u_sq * v_x.squeeze() + v_sq * v_y.squeeze())

        u = u + dt_true * RHS_u.unsqueeze(0).unsqueeze(0)
        v = v + dt_true * RHS_v.unsqueeze(0).unsqueeze(0)
        
        if t_step in train_steps:
            t_idx = np.where(train_steps == t_step)[0][0]
            u_series_true[t_idx] = u.squeeze()
            v_series_true[t_idx] = v.squeeze()
            
        if (t_step) % (Nt_true // 10 if Nt_true >= 10 else 1) == 0:
            print(f"完成 {t_step}/{Nt_true} 步 (t={t0 + t_step*dt_true:.4f})", end='\r')

    print("\nBurgers' 2D 模拟求解完成。")
    
    # --- 6. 降采样和加噪 ---
    u_true_np = u_series_true.cpu().numpy()
    v_true_np = v_series_true.cpu().numpy()

    # 降采样到输出网格 (Nt+1, Ny, Nx)
    u_train = u_true_np[:, idx_y, :][:, :, idx_x]
    v_train = v_true_np[:, idx_y, :][:, :, idx_x]
    
    # 加噪
    M_u = np.max(np.abs(u_train))
    M_v = np.max(np.abs(v_train))
    W_u = np.random.randn(*u_train.shape).astype(np.float32)
    W_v = np.random.randn(*v_train.shape).astype(np.float32)
    
    u_noisy = u_train + noise_level * M_u * W_u
    v_noisy = v_train + noise_level * M_v * W_v
    
    # 合并 u 和 v 变量 (Nt+1, Ny, Nx, 2)
    # solution_data = np.stack([u_noisy, v_noisy], axis=-1)
    solution_data = np.stack([u_noisy, v_noisy], axis=1)
    
    # --- 7. 组装输出字典 ---
    final_shape = solution_data.shape

    result = {
        'solution': solution_data,         # (Nt+1, Ny, Nx, 2)
        'x_solution': u_noisy,             # (Nt+1, Ny, Nx)
        'y_solution': v_noisy,             # (Nt+1, Ny, Nx)
        'x': x_train,
        'y': y_train,
        't_eval': t_train,
        'parameters': {
            'Lx': Lx, 'Ly': Ly, 't0': t0, 't_end': t_end,
            'nu': nu, 'Nx': Nx, 'Ny': Ny, 'Nt': Nt,
            'noise_level': noise_level, 'seed': seed,
            'Nx_true': Nx_true, 'Ny_true': Ny_true, 'Nt_true': Nt_true,
            'dt_true': dt_true,
            'discretization': 'Explicit Euler + 2nd Order Central Difference (Periodic BC)',
            'PDE_ref': '2D Burgers Equation'
        }
    }

    # 可选：保存到NPZ文件
    if save_path is not None:
        np.savez(save_path, **result)
        print(f"结果已保存至 {save_path}")

    print(f"Solution shape: {final_shape}")
    return result

def generate_2d_poisson_analytical(Lx=1.0, Ly=1.0, Nx=128, Ny=128, save_path=None):
    """
    生成二维泊松方程解析解的数值数据。

    该函数计算二维泊松方程 (Δu = f) 在矩形域 [0, Lx] × [0, Ly] 上，带有零 Dirichlet 边界条件
    (u=0 on boundaries) 的解析解数据。

    具体解析解公式：
    
    $$ u(x, y) = \sin\left( \frac{\pi x}{L_x} \right) \sin\left( \frac{\pi y}{L_y} \right) $$

    源项 f(x, y) 为：
    
    $$ f(x, y) = \Delta u = -\pi^2 \left( \frac{1}{L_x^2} + \frac{1}{L_y^2} \right) u(x, y) $$

    参数:
    - Lx, Ly: float, 空间域长度 [0, Lx] x [0, Ly]（默认1.0）。
    - Nx, Ny: int, 空间离散点数（默认128）。
    - save_path: str, 可选NPZ保存路径（默认None，不保存）。

    返回:
    - dict: {'solution': u_analytical (Ny, Nx), 'source': f_analytical (Ny, Nx), 'x': x_array, 'y': y_array}
    """
    
    # 1. 生成网格
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)  # 2D 空间网格，形状 (Ny, Nx)

    # 2. 定义解析解 u(x, y)
    def analytical_solution_u(X, Y, Lx, Ly):
        return np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)

    # 3. 计算解 u (形状: (Ny, Nx))
    u_analytical = analytical_solution_u(X, Y, Lx, Ly)

    # 4. 计算对应的源项 f(x, y) = Δu
    # Δu = u_xx + u_yy
    # u_xx = - (pi/Lx)^2 * u
    # u_yy = - (pi/Ly)^2 * u
    
    coeff = - (np.pi/Lx)**2 - (np.pi/Ly)**2
    f_analytical = coeff * u_analytical # 形状: (Ny, Nx)
    
    # 5. 组装输出字典
    result = {
        'solution': u_analytical,
        'source': f_analytical,
        'x': x,
        'y': y,
        'parameters': {
            'Lx': Lx,
            'Ly': Ly,
            'Nx': Nx,
            'Ny': Ny
        }
    }

    # 可选：保存到NPZ文件
    if save_path is not None:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        np.savez(save_path, **result)
        print(f"结果已保存至 {save_path}")

    print(f"Solution shape (u): {u_analytical.shape}")
    print(f"Source term shape (f): {f_analytical.shape}")
    return result

def generate_2d_convection_diffusion_analytical(
    Lx=1.0, Ly=1.0, t0=0.0, t_end=1.0, 
    Nx=128, Ny=128, Nt=100, 
    nu=0.01, cx=0.5, cy=0.0, sigma0 = 0.05,
    save_path=None
):
    """
    生成二维对流-扩散方程解析解（移动的高斯波包）。

    解析解基于无界域高斯解：
    u(x, y, t) = A(t) * exp( - ( (x - x_c(t))^2 + (y - y_c(t))^2 ) / (4 * nu * (t + t_start)) )
    
    其中中心点 (x_c, y_c) 随时间以速度 (cx, cy) 移动。

    参数:
    - nu: 扩散系数 (建议 0.001 - 0.01 以保持波形)
    - cx, cy: 对流速度 (建议 cx=0.5, cy=0 以观察明显的横向移动)
    
    - sigma0:初始方差 (决定波包的宽度). 越小，波包越尖锐。为了避免接触边界，设得小一点。
    """
    
    # 1. 生成网格
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    t_eval = np.linspace(t0, t_end, Nt + 1) 
    X, Y = np.meshgrid(x, y) # 形状 (Ny, Nx)

    # 2. 定义高斯波包参数
    # 初始中心位置 (x0, y0)
    # 我们把它放在偏左的位置 (0.25, 0.5)，这样它有空间向右移动
    x0 = 0.25 * Lx
    y0 = 0.5 * Ly
    
    
    # 定义解析解函数
    def gaussian_packet(X, Y, t, cx, cy, nu, x0, y0, sigma0):
        """
        X, Y: 网格
        t: 当前时间
        cx, cy: 速度
        nu: 扩散系数
        x0, y0: 初始位置
        sigma0: 初始宽度的标准差参数
        """
        # 计算随时间变化的中心位置 (对流效应)
        x_center = x0 + cx * t
        y_center = y0 + cy * t
        
        # 计算随时间变化的扩散宽度 (扩散效应)
        # 随着时间 t 增加，分母变大，波包变宽，峰值变低
        # 这里的公式源自热核的基本解
        sigma_sq_t = sigma0**2 + 2 * nu * t
        
        # 计算高斯分布
        # 为了数值稳定性，不使用 1/(4*pi*nu*t) 这种归一化系数，而是保持峰值在一个合理范围
        # 我们只关注形状的演化
        exponent = -((X - x_center)**2 + (Y - y_center)**2) / (2 * sigma_sq_t)
        u = np.exp(exponent)
        
        return u

    # 3. 计算所有时间步的解
    u_analytical = np.zeros((Nt + 1, Ny, Nx))
    
    print(f"Generating data with: nu={nu}, cx={cx}, cy={cy}")
    print(f"Initial Center: ({x0}, {y0}) -> Expected Final Center: ({x0 + cx*t_end}, {y0 + cy*t_end})")

    for i, t in enumerate(t_eval):
        u_analytical[i] = gaussian_packet(X, Y, t, cx, cy, nu, x0, y0, sigma0)

    # 4. 组装输出
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
        print(f"结果已保存至 {save_path}")

    return result

# 生成数据并可视化
if __name__ == "__main__":
    # 参数设置
    Lx, Ly = 1.0, 1.0
    Nx, Ny = 32, 32
    Nt = 120

    # 二维波动方程解析解
    # wave_data = generate_2d_wave_analytical(
    #     Lx=Lx, Ly=Ly, 
    #     t0=0, t_end=2.4, 
    #     Nx=Nx, Ny=Ny, 
    #     Nt=Nt, 
    #     c=1.0, 
    #     save_path="data/wave_solution_2d(32).npz"  # 可选保存
    # )
    # wave_u = wave_data['solution']
    # wave_t = wave_data['t_eval']
    # create_interactive_wave_viz(wave_u, wave_t, Lx, Ly, interval=0.05)

    # 生成二维纳维-斯托克斯 Taylor-Green 涡旋
    ns_data = generate_2d_naviers_stokes(
        Lx=2 * np.pi, Ly=2 * np.pi, 
        t0=0, t_end=12.0, 
        Nx=Nx, Ny=Ny, Nt=Nt,
        nu=0.05, U_max=1.0, seed=42,
        save_path="data/naviers_stokes_2d(32).npz"
        )
    create_interactive_wave_viz(ns_data['x_solution'], ns_data['t_eval'], 2*np.pi, 2*np.pi, interval=0.1)


    # # 二维对流扩散方程解析解
    wave_data = generate_2d_convection_diffusion_analytical(
        Lx=Lx, Ly=Ly, 
        t0=0, t_end=2.4, 
        Nx=Nx, Ny=Ny, 
        Nt=Nt, 
        cy=0.09,cx=0.18,nu=0.002,
        sigma0=0.1,
        save_path="data/convection_diffusion_2d(32).npz"  # 可选保存
    )
    wave_u = wave_data['solution']
    wave_t = wave_data['t_eval']
    create_interactive_wave_viz(wave_u, wave_t, Lx, Ly, interval=0.05)
