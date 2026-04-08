import os
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback

from hls_env import HLSSchedulerEnv 
from gnn_extractor import GNNFeaturesExtractor

def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Extracts the action mask from the environment's observation space.
    This tells MaskablePPO which nodes are legal to schedule.
    """
    if hasattr(env, "unwrapped"):
        return env.unwrapped.action_masks()
    return env.action_masks()

def make_env():
    """Initializes the environment and wraps it with the ActionMasker."""
    # Load kernel.py as source_code to compile the graph
    try:
        with open("kernel.py", "r") as f:
            source_code = f.read()
    except Exception as e:
        print("Could not read kernel.py, using default fallback.")
        source_code = None
        
    env = HLSSchedulerEnv(source_code=source_code)
    # Wrap for ML observability masking
    env = ActionMasker(env, mask_fn)
    return env

class SaveBestMakeSpanCallback(BaseCallback):
    """
    Custom callback to monitor MakeSpan (total cycles) and save the best model.
    """
    def __init__(self, save_path: str, verbose=1):
        super(SaveBestMakeSpanCallback, self).__init__(verbose)
        self.save_path = save_path
        self.best_makespan = float('inf')

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if the episode just ended
        if self.locals["dones"][0]:
            # Retrieve the final cycle count from the environment's info dict
            info = self.locals["infos"][0]
            if "hls_state" in info:
                # The makespan is exactly the current cycle at termination!
                current_makespan = info["hls_state"].current_cycle
                
                # Log to TensorBoard
                self.logger.record("hls/makespan", current_makespan)

                # Save if it's a new record
                if current_makespan < self.best_makespan:
                    self.best_makespan = current_makespan
                    if self.verbose > 0:
                        print(f"\nNew best MakeSpan achieved: {self.best_makespan} cycles! Saving model...")
                    self.model.save(os.path.join(self.save_path, "best_hls_model_phase3"))
        return True

if __name__ == "__main__":
    print("--- Initializing HLS Training Environment ---")
    
    # Setup directories
    log_dir = "./hls_tensorboard/"
    save_dir = "./models/"
    
    env = make_env()

    # Initialize MaskablePPO with the GNN Extractor
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": GNNFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch": [256, 256]
        },
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=5e-4, # Optimized LR as per final checklist
        gamma=0.99          # High discount factor for long-term critical path planning
    )

    # Execute Training
    callback = SaveBestMakeSpanCallback(save_path=save_dir)
    print("--- Starting Training: 200,000 Timesteps ---")
    
    model.learn(total_timesteps=200000, callback=callback)
    
    print("--- Training Complete ---")
    model.save(os.path.join(save_dir, "final_hls_model_phase3"))
