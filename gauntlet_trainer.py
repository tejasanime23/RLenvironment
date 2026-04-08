import os
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from hls_env import HLSSchedulerEnv
from gnn_extractor import GNNFeaturesExtractor
from gauntlet_wrapper import GauntletWrapper

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()

def make_gauntlet_env(max_alu=99, max_mac=99, max_mem=99):
    """Initializes the gauntlet environment with the specific constraints."""
    # Base environment with hardening
    env = HLSSchedulerEnv(
        source_code=None, # Loaded by wrapper
        max_alu=max_alu, 
        max_mac=max_mac, 
        max_mem=max_mem,
        universal_max_nodes=300
    )
    # Apply the Multi-Kernel Gauntlet Shuffle
    env = GauntletWrapper(env, benchmarks_dir="benchmarks")
    # Wrap for ML observability masking
    env = ActionMasker(env, mask_fn)
    return env

class GauntletCallback(BaseCallback):
    """Callback for monitoring multi-kernel performance."""
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
                self.logger.record(f"gauntlet/{self.prefix}_makespan", current_makespan)
                
                # We save based on any improvement in ANY kernel during the gauntlet
                if current_makespan < self.best_makespan and current_makespan > 0:
                    self.best_makespan = current_makespan
                    self.model.save(os.path.join(self.save_path, f"best_gauntlet_model_{self.prefix}"))
        return True

if __name__ == "__main__":
    log_dir = "./hls_tensorboard/"
    save_dir = "./models/"
    
    print("\n[GAUNTLET] Phase 1: Unlimited Topology Mastery (Discovery)")
    env_unconstrained = make_gauntlet_env(max_alu=10, max_mac=10, max_mem=10)
    
    # 8GB Memory Wall Protection: buffer_size=20000, n_steps=1024
    model = MaskablePPO(
        "MultiInputPolicy",
        env_unconstrained,
        policy_kwargs=dict(
            features_extractor_class=GNNFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=[256, 256]
        ),
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=5e-4, # Aggressive generalization
        gamma=0.99,
        n_steps=1024,        # Flush gradients frequently
        batch_size=64,      # Memory safe batches
        # buffer_size=20000 # Only applies to Off-Policy like DQN/SAC, PPO is On-Policy
    )
    
    cb_phase1 = GauntletCallback(prefix="discovery", save_path=save_dir)
    model.learn(total_timesteps=40000, callback=cb_phase1)
    
    print("\n[GAUNTLET] Phase 2: Restricted Hardware Mastery (Pro-League)")
    env_constrained = make_gauntlet_env(max_alu=4, max_mac=2, max_mem=2)
    model.set_env(env_constrained)
    model.learning_rate = 1e-4 # Fine-tuning LR
    
    cb_phase2 = GauntletCallback(prefix="performance", save_path=save_dir)
    model.learn(total_timesteps=60000, callback=cb_phase2, reset_num_timesteps=False)
    
    model.save(os.path.join(save_dir, "ultimate_gauntlet_agent"))
    print("\n[GAUNTLET] Mission Accomplished. Universal Scheduler Trained.")
