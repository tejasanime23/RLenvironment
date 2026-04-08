import numpy as np
from hls_env import HLSSchedulerEnv

def run_simulation(seed, source_code):
    env = HLSSchedulerEnv(source_code=source_code)
    obs, info = env.reset(seed=seed)
    
    total_reward = 0.0
    done = False
    
    # Use a fixed sequence of random actions for the determinism check
    # We use a local RNG to ensure the action sequence itself is seeded
    rng = np.random.RandomState(seed)
    
    while not done:
        # Get action mask
        mask = env.action_masks()
        valid_actions = np.where(mask == 1)[0]
        
        # Pick a valid action deterministically based on our local RNG
        action = rng.choice(valid_actions)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
    return {
        "cycles": info["hls_state"].current_cycle,
        "reward": total_reward,
        "area": info["area"]
    }

if __name__ == "__main__":
    print("--- Starting Determinism & Variance Audit ---")
    
    with open("benchmarks/level_1_warmup.py", "r") as f:
        source = f.read()

    results = []
    for i in range(5):
        # Run 5 times with the EXACT same seed
        res = run_simulation(seed=42, source_code=source)
        results.append(res)
        print(f"   Run {i+1}: Cycles={res['cycles']}, Reward={res['reward']:.4f}, Area={res['area']}")

    # Check Variance
    all_cycles = [r["cycles"] for r in results]
    all_rewards = [r["reward"] for r in results]
    
    variance_cycles = np.var(all_cycles)
    variance_rewards = np.var(all_rewards)
    
    print("-" * 40)
    if variance_cycles == 0 and variance_rewards == 0:
        print("✅ SUCCESS: Zero Variance detected across multiple runs!")
        print("   Score Determinism: 100%")
    else:
        print("❌ FAILURE: Non-deterministic behavior detected!")
        print(f"   Cycle Variance: {variance_cycles}")
        print(f"   Reward Variance: {variance_rewards}")
