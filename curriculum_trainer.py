import os
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from hls_env import HLSSchedulerEnv
from gnn_extractor import GNNFeaturesExtractor

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()

def make_env(max_alu=99, max_mac=99, max_mem=99):
    """Initializes the environment with the specific constraints given."""
    try:
        with open("kernel.py", "r") as f:
            source_code = f.read()
    except Exception as e:
        print("Could not read kernel.py")
        source_code = None
        
    env = HLSSchedulerEnv(
        source_code=source_code, 
        max_alu=max_alu, 
        max_mac=max_mac, 
        max_mem=max_mem
    )
    env = ActionMasker(env, mask_fn)
    return env

class CurriculumCallback(BaseCallback):
    """Callback designed to track phase-specific models seamlessly."""
    def __init__(self, prefix, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_makespan = float('inf')
        self.prefix = prefix
        
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]
            if "hls_state" in info:
                current_makespan = info["hls_state"].current_cycle
                self.logger.record(f"hls_{self.prefix}/makespan", current_makespan)
                
                if "area" in info:
                    self.logger.record(f"hls_{self.prefix}/final_area", info["area"])
                
                if current_makespan < self.best_makespan and current_makespan > 0:
                    self.best_makespan = current_makespan
                    if self.verbose > 0:
                        print(f"\n[{self.prefix}] New Best MakeSpan (within bounds): {self.best_makespan} cycles! Saving...")
                    self.model.save(os.path.join(self.save_path, f"{self.prefix}_best_model"))
        return True

if __name__ == "__main__":
    log_dir = "./hls_tensorboard/"
    save_dir = "./models/"
    
    print("\n=======================================================")
    print("=== PHASE 1: Topology Training (Unlimited Hardware) ===")
    print("=======================================================\n")
    
    # 1. Start Wide and Unlimited
    env_unconstrained = make_env(max_alu=99, max_mac=99, max_mem=99)
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env_unconstrained,
        policy_kwargs=dict(
            features_extractor_class=GNNFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=[256, 256]
        ),
        verbose=0,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        gamma=0.99
    )
    
    cb_phase1 = CurriculumCallback(prefix="phase1_unlimited", save_path=save_dir)
    # Train heavily on sequence rules
    model.learn(total_timesteps=30000, callback=cb_phase1)
    print(f"\n[Phase 1 Complete] The Agent locked the topology limit at {cb_phase1.best_makespan} cycles!")
    
    
    print("\n=======================================================")
    print("=== PHASE 2: Dropping the Hammer (Resource Limits)  ===")
    print("=======================================================\n")
    
    # 2. Swap the strict Phase 3 limits into a fresh clone
    env_constrained = make_env(max_alu=2, max_mac=1, max_mem=1)
    
    # Crucial Step: Transfer the existing brain into the constrained environment!
    model.set_env(env_constrained)
    
    # Optional RL Trick: Drop learning rate to prevent violent forgetting (Catastrophic Forgetting)
    model.learning_rate = 1e-4
    
    cb_phase2 = CurriculumCallback(prefix="phase2_constrained", save_path=save_dir)
    
    # Resume training using identical weights natively
    model.learn(total_timesteps=50000, callback=cb_phase2, reset_num_timesteps=False)
    
    print(f"\n[Phase 2 Complete] Final bottlenecked constraint handling optimized to {cb_phase2.best_makespan} cycles!")
    
    model.save(os.path.join(save_dir, "ultimate_RL_scheduler"))
    print("Agent fully trained! Saved to models/ultimate_RL_scheduler")
