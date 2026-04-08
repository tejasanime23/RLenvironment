import argparse
import random
import numpy as np
from hls_env import HLSSchedulerEnv

def main():
    parser = argparse.ArgumentParser(description="Run Random Agent Baseline on HLS Environment.")
    parser.add_argument("kernel_file", type=str, nargs='?', default=None, help="Path to python kernel file")
    args = parser.parse_args()

    source_code = None
    if args.kernel_file:
        try:
            with open(args.kernel_file, "r") as f:
                source_code = f.read()
            print(f"Loaded source code from {args.kernel_file}")
        except FileNotFoundError:
            print(f"Error: Could not find file {args.kernel_file}")
            return
            
    # Initialize the Environment
    env = HLSSchedulerEnv(source_code=source_code)
    
    print("\n--- Starting Random Agent Baseline ---")
    print(f"Environment Max Nodes: {env.max_nodes}")
    print("Action 0 to N-1 are Nodes, Action N is WAIT.")
    
    obs, info = env.reset()
    
    terminated = False
    truncated = False
    total_reward = 0
    steps = 0
    
    while not (terminated or truncated):
        steps += 1
        
        # Retrieve the legal actions mask from the observation
        # mask is shape (max_nodes + 1,)
        # 1 means valid, 0 means invalid
        action_mask = obs["action_mask"]
        
        # Find indices of all valid actions
        valid_actions = np.where(action_mask == 1)[0]
        
        if len(valid_actions) == 0:
            print("Error: No valid actions found (Deadlock state!)")
            break
            
        # Select a random legal action
        # If possible, we might want to prioritize scheduling over waiting
        # In a purely random agent, we just pick from valid_actions
        action = random.choice(valid_actions)
        
        # Optional heuristic for random baseline: if there are ready nodes, pick them 80% of the time instead of WAIT
        # (Since random waiting might artificially bloat cycle count)
        # We can just leave it fully uniform random for now.

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # For debugging, let's print scheduling actions (skip generic WAIT when no actual log needed)
        if action != env.BUTTON_ACTION:
            # Action could be a pragma or a schedule, decode it:
            node_id = action // env.PRAGMAS
            pragma = action % env.PRAGMAS
            if env.unwrapped.phase == "TRANSFORM":
                print(f"Step {steps:3d} | Applied Pragma {pragma} to Node {node_id}")
            else:
                print(f"Step {steps:3d} | Scheduled Node {node_id} | Current Cycle: {info['hls_state'].current_cycle}")

    # Final stats
    final_cycle = info['hls_state'].current_cycle
    print("\n--- Baseline Run Completed ---")
    print(f"Total Steps Taken: {steps}")
    print(f"Total MakeSpan (Clock Cycles): {final_cycle}")
    print(f"Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
