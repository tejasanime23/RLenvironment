import os
import argparse
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from curriculum_trainer import make_env

def plot_gantt_chart(history, env):
    # 1. Post-Processing: Resource Binding
    # Dynamically generate hardware IDs based on what the environment legally unlocked!
    hw = env.unwrapped.hls_state.hardware
    resource_free_time = {}
    
    for i in range(hw.total_alu): resource_free_time[f"ALU_{i}"] = 0
    for i in range(hw.total_mac): resource_free_time[f"MAC_{i}"] = 0
    for i in range(hw.total_mem): resource_free_time[f"MEM_{i}"] = 0
    
    bound_history = []
    
    # Sort history chronologically just to be definitively safe for binding checks
    history = sorted(history, key=lambda x: x["start_cycle"])
    
    for task in history:
        res_type = task["resource"] # 'ALU', 'MAC', or 'MEM'
        start = task["start_cycle"]
        
        assigned_id = None
        for r_id in resource_free_time.keys():
            if r_id.startswith(res_type) and resource_free_time[r_id] <= start:
                assigned_id = r_id
                resource_free_time[r_id] = task["end_cycle"]
                break
                
        if assigned_id is None:
            raise ValueError(f"Constraint Violation Detected for Node {task['node_id']} ({res_type}) at cycle {start}!")
            
        task["physical_id"] = assigned_id
        bound_history.append(task)

    # 2. Matplotlib broken_barh setup
    fig, ax = plt.subplots(figsize=(14, max(7, len(resource_free_time)*0.8)))
    
    y_mapping = {}
    y_ticks = []
    y_labels = []
    
    current_y = 10
    colors = {"ALU": "tab:blue", "MAC": "tab:orange", "MEM": "tab:green"}
    
    # Sort keys for consistent plotting from bottom to top
    for key in sorted(resource_free_time.keys(), reverse=True):
        y_mapping[key] = current_y
        y_ticks.append(current_y)
        y_labels.append(key)
        current_y += 10
    
    for task in bound_history:
        y_pos = y_mapping[task["physical_id"]]
        duration = task["end_cycle"] - task["start_cycle"]
        color = colors.get(task["resource"], "gray")
        
        ax.broken_barh([(task["start_cycle"], duration)], (y_pos - 4, 8), facecolors=color, edgecolor='black')
        
        ax.text(task["start_cycle"] + duration/2, y_pos, f"{task['op']}\nN{task['node_id']}", 
                ha='center', va='center', color='white', fontsize=8, fontweight='bold')

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Clock Cycles')
    ax.set_title('AI-Optimized HLS Hardware Schedule (Task 3: Dynamic Pragmas!)')
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('schedule_gantt.png', dpi=300)
    print("Saved Gantt Chart to schedule_gantt.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL Agent Schedule")
    parser.add_argument("kernel_file", type=str, nargs='?', default=None, help="Path to python kernel file")
    args = parser.parse_args()
    
    print("--- Rolling out Task 3 (Pragma Transformers) ---")
    
    # We load the Phase 1 checkpoint since the script was interrupted before Phase 2 could finish and overwrite ultimate_RL_scheduler
    model_path = os.path.join("models", "phase1_unlimited_best_model")
    model = MaskablePPO.load(model_path)
    
    source_code = None
    if args.kernel_file:
        try:
            with open(args.kernel_file, "r") as f:
                source_code = f.read()
            print(f"Loaded source code from {args.kernel_file}")
        except FileNotFoundError:
            print(f"Error: Could not find file {args.kernel_file}")
            exit(1)
            
    # Rather than make_env which hardcodes params, we instantiate it natively to pass source_code
    from hls_env import HLSSchedulerEnv
    from stable_baselines3.common.monitor import Monitor
    from sb3_contrib.common.wrappers import ActionMasker
    
    def mask_fn(env): return env.action_masks()
    
    base_env = HLSSchedulerEnv(source_code=source_code, max_alu=2, max_mac=1, max_mem=1)
    env = ActionMasker(Monitor(base_env), mask_fn)
    
    obs, info = env.reset()
    
    done = False
    while not done:
        # MASKABLE PPO INFERENCE REQUIRES THE ACTION MASK EXPLICITLY PASSED!
        action, _states = model.predict(obs, action_masks=obs["action_mask"], deterministic=True) 
        
        # SB3 predict returns a numpy array, we need a standard scalar
        action_scalar = int(action)
        
        obs, reward, terminated, truncated, info = env.step(action_scalar)
        done = terminated or truncated

    makespan = info["hls_state"].current_cycle
    print(f"\nDeterministic MakeSpan Bound: {makespan} cycles")
    
    history = env.unwrapped.execution_history
    
    try:
        plot_gantt_chart(history, env)
    except Exception as e:
        print(f"Failed configuring plot mathematically: {e}")
