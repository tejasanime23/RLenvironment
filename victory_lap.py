import os
import gymnasium as gym
import numpy as np
import glob
import networkx as nx
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from hls_env import HLSSchedulerEnv
from gauntlet_trainer import make_gauntlet_env

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()

def validate_agent():
    model_path = "./models/ultimate_gauntlet_agent.zip"
    if not os.path.exists(model_path):
        print("Error: ultimate_gauntlet_agent.zip not found!")
        return

    print(f"\n[VICTORY LAP] Loading Agent: {model_path}")
    model = MaskablePPO.load(model_path)
    
    kernels = glob.glob("benchmarks/gauntlet_*.py")
    if os.path.exists("kernel.py"):
        kernels.append("kernel.py")
        
    print("-" * 75)
    print(f"| {'Kernel Name':<25} | {'Cycles':<8} | {'Nodes':<6} | {'Status':<25} |")
    print("-" * 75)
    
    for k_path in sorted(kernels):
        try:
            # Use raw environment to avoid randomized shuffling during validation
            with open(k_path, "r") as f:
                code = f.read()
                
            env = HLSSchedulerEnv(
                source_code=code,
                max_alu=4,
                max_mac=2,
                max_mem=2,
                universal_max_nodes=300
            )
            env = ActionMasker(env, mask_fn)
            obs, info = env.reset()
            
            # Run inference
            done = False
            while not done:
                # Get action mask
                mask = env.unwrapped.action_masks()
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            cycles = info["hls_state"].current_cycle
            nodes = len(info["hls_state"].nodes)
            status = "SUCCESS" if not getattr(env.unwrapped, "synthesis_error", False) else "REJECTED (Safe)"
            
            print(f"| {os.path.basename(k_path):<25} | {cycles:<8} | {nodes:<6} | {status:<25} |")
        except Exception as e:
            print(f"| {os.path.basename(k_path):<25} | ERROR    | N/A    | {str(e)[:23]:<25} |")

    print("-" * 75)
    print("\nUniversal Hardware Scheduler Verification Complete.")

if __name__ == "__main__":
    validate_agent()
