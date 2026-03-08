import numpy as np
import os

def add_noise_to_file(file_path, noise_levels=[0.01, 0.05, 0.10], seed=42):
    """
    Read NPZ file and generate new files with different noise levels.
    
    Args:
        file_path (str): Original .npz file path
        noise_levels (list): List of noise ratios, e.g., [0.01] represents 1% noise
        seed (int): Random seed
    """
    if not os.path.exists(file_path):
        print(f"[Skip] File not found: {file_path}")
        return

    print(f"\nProcessing: {file_path}")
    
    # 1. Load data
    # allow_pickle=True is used to prevent loading errors for objects saved by older numpy versions
    data_dict = np.load(file_path, allow_pickle=True)
    
    # Convert npz object to a mutable dict
    original_data = {key: data_dict[key] for key in data_dict.files}
    
    # 2. Identify which keys are physical fields that require noise injection
    # Heuristic rules:
    # - Typically coordinates (x, y, t) are 1D arrays
    # - Physical fields (u, v, solution) are usually 2D or 3D arrays (Nt, Ny, Nx)
    # - Can also be hard-coded based on key names
    field_keys = []
    for key, val in original_data.items():
        # Exclude common coordinate names
        if key in ['x', 'y', 't', 't_eval', 'dt', 'dx', 'dy']:
            continue
        # Exclude scalars or 1D arrays (usually parameters or coordinates)
        if val.ndim <= 1:
            continue
        
        field_keys.append(key)
    
    print(f"  -> Identified fields to noise: {field_keys}")

    # 3. Loop to generate data with different noise levels
    for nl in noise_levels:
        # Set random seed (ensure deterministic noise patterns for each noise level)
        # Use seed + noise_level * 100 to ensure different patterns across levels, but consistency for a fixed level
        np.random.seed(seed + int(nl * 100))
        
        noisy_data_dict = original_data.copy()
        
        for key in field_keys:
            clean_signal = original_data[key]
            
            # Calculate standard deviation of the signal (to measure signal magnitude)
            # If the signal is all zeros (extremely rare), std will be 0, and no noise is added
            sig_std = np.std(clean_signal)
            
            # Generate Gaussian noise
            # Noise = level * std * N(0,1)
            noise = nl * sig_std * np.random.randn(*clean_signal.shape)
            
            # Superimpose noise
            noisy_signal = clean_signal + noise
            
            # Save into dictionary
            noisy_data_dict[key] = noisy_signal.astype(np.float32) # Keep as float32 to save space
            
            # Print statistics for inspection
            # print(f"    [{key}] Noise Level {nl*100}%: Signal Std={sig_std:.4f}, Noise Std={np.std(noise):.4f}")

        # 4. Save file
        # Naming format: original_name_noise_0.01.npz
        dir_name, base_name = os.path.split(file_path)
        name_root, ext = os.path.splitext(base_name)
        new_filename = f"{name_root}_noise_{nl}{ext}"
        save_path = os.path.join(dir_name, new_filename)
        
        np.savez(save_path, **noisy_data_dict)
        print(f"  -> Saved: {save_path}")

def main():
    # Define the list of target data files
    target_files = [
        "data/wave_solution_2d(32).npz",
        "data/convection_diffusion_2d(32).npz",
        "data/naviers_stokes_2d(32).npz"
    ]
    
    # Define noise levels: 0.1%, 0.5%, 1%, 2%
    # 0.01 means the noise standard deviation is 1% of the data standard deviation
    # levels = [0.001, 0.005, 0.01, 0.02, 0.05]
    levels = [0.1]
    
    for f in target_files:
        add_noise_to_file(f, noise_levels=levels)

if __name__ == "__main__":
    main()