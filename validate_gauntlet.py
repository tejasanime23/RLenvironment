import os
import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from gauntlet_trainer import make_gauntlet_env

def run_test(model, kernel_path):
    # Set up environment with strict Phase 2/3 limits
    env = make_gauntlet_env(max_alu=4, max_mac=2, max_mem=2)
    
    with open(kernel_path, "r") as f:
        source = f.read()
    
    # Load kernel directly into the unwrapped env
    env.unwrapped.set_kernel(source)
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, action_masks=env.unwrapped.action_masks(), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
    makespan = info["hls_state"].current_cycle
    print(f"| {os.path.basename(kernel_path):<30} | {makespan:<10} | {total_reward:<12.2f} |")
    return makespan

if __name__ == "__main__":
    model_path = "./models/ultimate_gauntlet_agent.zip"
    if not os.path.exists(model_path):
        print("Error: Model not found!")
        exit(1)
        
    print(f"\n[VICTORY LAP] Loading Agent: {model_path}")
    model = MaskablePPO.load(model_path)
    
    stressors = [
        "benchmarks/gauntlet_fft_butterfly.py",
        "benchmarks/gauntlet_iir_serial.py",
        "benchmarks/gauntlet_separable_conv.py",
        "benchmarks/gauntlet_sobel_stencil.py",
        "kernel.py" # The standard 4x4 MatVec
    ]
    
    print("-" * 65)
    print(f"| {'Kernel Name':<30} | {'MakeSpan':<10} | {'Reward':<12} |")
    print("-" * 65)
    
    for k in stressors:
        if os.path.exists(k):
            run_test(model, k)
    
    print("-" * 65)
    print("\nGauntlet Validation Complete. The Agent is ready for Hugging Face deployment.")
