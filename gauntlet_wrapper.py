import os
import random
import glob
import numpy as np
import gymnasium as gym

class GauntletWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that manages a pool of hardware kernels.
    On every reset, it hot-swaps the underlying DAG to force generalization.
    Supports a 'Trap Kernel' to verify environment robustness.
    """
    def __init__(self, env, benchmarks_dir="benchmarks"):
        super(GauntletWrapper, self).__init__(env)
        self.benchmarks_dir = benchmarks_dir
        
        # Discover all valid gauntlet kernels
        self.math_kernels = glob.glob(os.path.join(benchmarks_dir, "gauntlet_*.py"))
        # Exclude the trap kernel from the math pool
        self.math_kernels = [k for k in self.math_kernels if "trap" not in k]
        
        self.trap_kernel = os.path.join(benchmarks_dir, "gauntlet_trap_dynamic.py")
        
        # Add the default 4x4 mat-vec to the pool
        if os.path.exists("kernel.py"):
            self.math_kernels.append("kernel.py")
            
        print(f"[GAUNTLET] Initialized with {len(self.math_kernels)} math kernels and 1 trap kernel.")

    def reset(self, seed=None, options=None):
        # 1. Determine kernel type: 10% chance for the Trap
        if random.random() < 0.10 and os.path.exists(self.trap_kernel):
            selected_kernel = self.trap_kernel
            status = "TRAP"
        else:
            selected_kernel = random.choice(self.math_kernels)
            status = "MATH"
            
        # 2. Hot-swap the kernel in the base environment
        try:
            with open(selected_kernel, "r") as f:
                source = f.read()
            
            # Use our new hot-swap method
            self.env.unwrapped.set_kernel(source)
            
        except Exception as e:
            print(f"[GAUNTLET ERROR]: Failed to load {selected_kernel}: {e}")
            
        # 3. Perform standard reset
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Inject Gauntlet info for the agent/Nemotron to reason about
        info["gauntlet_status"] = status
        info["kernel_name"] = os.path.basename(selected_kernel)
        
        # Update status string for Nemotron reasoning
        current_status = info.get("status", "")
        info["status"] = f"Gauntlet Mode: {status} | Kernel: {info['kernel_name']} | {current_status}"
        
        return obs, info

    def step(self, action):
        # If the environment detected a synthesis error during reset, 
        # any action should immediately terminate with a penalty.
        if getattr(self.env.unwrapped, "synthesis_error", False):
            obs = self.env.unwrapped._get_obs()
            info = self.env.unwrapped._get_info()
            info["status"] = "[SYNTHESIS REJECTED]: This kernel contains dynamic logic and cannot be mapped to static hardware."
            # Failure state
            return obs, -100.0, True, False, info
            
        return self.env.step(action)
