import os
import argparse
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from hls_env import HLSSchedulerEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from server.graders import grade_task_1, grade_task_2, grade_task_3
from compiler import HLSSynthesisError

def extract_global_state(obs, info):
    """ Helper to format info mimicking the OpenEnv API response """
    # Graders expect a dict with 'global_state'
    global_s = obs.get('global_state', [info['hls_state'].current_cycle] * 5)
    meta = info.get('metadata', {"critical_path_depth": 18, "total_ops": 177, "max_area": 1000.0})
    return {"global_state": global_s, "metadata": meta}

def plot_multi_gantt_chart(history, env, filename, title_str):
    hw = env.unwrapped.hls_state.hardware
    resource_free_time = {}
    
    for i in range(hw.total_alu): resource_free_time[f"ALU_{i}"] = 0
    for i in range(hw.total_mac): resource_free_time[f"MAC_{i}"] = 0
    for i in range(hw.total_mem): resource_free_time[f"MEM_{i}"] = 0
    
    bound_history = []
    history = sorted(history, key=lambda x: x["start_cycle"])
    
    for task in history:
        res_type = task["resource"]
        start = task["start_cycle"]
        assigned_id = None
        for r_id in resource_free_time.keys():
            if r_id.startswith(res_type) and resource_free_time[r_id] <= start:
                assigned_id = r_id
                resource_free_time[r_id] = task["end_cycle"]
                break
        if assigned_id is None:
            # If bound failed, just map generically for the visualizer
            assigned_id = f"{res_type}_0" 
            
        task["physical_id"] = assigned_id
        bound_history.append(task)
            
    # 2. Filter to only ACTIVE resources (prevents 999 empty rows in Task 1)
    active_resources = list(set(task["physical_id"] for task in bound_history))
    
    # Natural sort: ALU_0, ALU_1, ..., ALU_10, MAC_0...
    def resource_sort_key(s):
        import re
        parts = re.split('([0-9]+)', s)
        return [int(p) if p.isdigit() else p for p in parts]
    
    active_resources = sorted(active_resources, key=resource_sort_key, reverse=True)
    
    num_rows = len(active_resources)
    row_h = 0.5 if num_rows > 20 else 0.8
    dpi_val = 150 if num_rows > 30 else 300
    
    fig, ax = plt.subplots(figsize=(14, max(4, num_rows * row_h)))
    y_mapping, y_ticks, y_labels = {}, [], []
    current_y = 10
    colors = {"ALU": "tab:blue", "MAC": "tab:orange", "MEM": "tab:green"}
    
    for key in active_resources:
        y_mapping[key] = current_y
        y_ticks.append(current_y)
        y_labels.append(key)
        current_y += 10
    
    for task in bound_history:
        y_pos = y_mapping[task["physical_id"]]
        duration = task["end_cycle"] - task["start_cycle"]
        color = colors.get(task["resource"], "gray")
        ax.broken_barh([(task["start_cycle"], duration)], (y_pos - 4, 8), facecolors=color, edgecolor='black')
        
        # Only draw text if we have enough vertical space
        if num_rows < 50:
            ax.text(task["start_cycle"] + duration/2, y_pos, f"{task['op']}\nN{task['node_id']}", 
                    ha='center', va='center', color='white', fontsize=7 if num_rows > 20 else 8, fontweight='bold')


    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Clock Cycles')
    ax.set_title(title_str)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi_val)
    print(f"   Generated Gantt Chart:   {filename} ({num_rows} resources)")
    plt.close(fig)

def run_evaluation(task_id, description, env_params, model_file, grader_func, source_code):
    print("=" * 80)
    print(f"🤖 EXECUTING: {task_id.upper()}")
    print(f"   {description}")
    print(f"   Hardware Constraints - ALU: {env_params['max_alu']} | MAC: {env_params['max_mac']} | MEM: {env_params['max_mem']}")
    print("=" * 80)

    model_path = os.path.join("models", model_file)
    if not os.path.exists(model_path + ".zip" if not model_path.endswith(".zip") else model_path):
        print(f"Model {model_path} not found. Skipping...")
        return
        
    model = MaskablePPO.load(model_path)
    
    def mask_fn(env): return env.action_masks()
    
    try:
        base_env = HLSSchedulerEnv(source_code=source_code, **env_params)
    except HLSSynthesisError as e:
        print(f"\n   ❌ [HARDWARE COMPILER REJECTED]:\n   {e}\n")
        print("-" * 80)
        return
        
    env = ActionMasker(Monitor(base_env), mask_fn)
    
    # Force Scheduling phase for Task 1/2, allow TRANSFORM for Task 3
    initial_phase = "SCHEDULE" if "TASK_3" not in task_id.upper() else "TRANSFORM"
    obs, info = env.reset(options={"initial_phase": initial_phase})
    
    steps = 0
    total_reward = 0.0
    done = False
    
    while not done:
        action, _ = model.predict(obs, action_masks=obs["action_mask"], deterministic=True)
        action_val = int(action)
        
        prev_area = info.get("area", 0)
        
        obs, reward, terminated, truncated, info = env.step(action_val)
        done = terminated or truncated
        total_reward += float(reward)
        steps += 1
        
        # Display Step Details beautifully
        status_msg = info.get("status", "")
        # print(f"   [ENV STATUS]: {status_msg}")
        
        is_button = (action_val == base_env.BUTTON_ACTION)
        
        if is_button:
            if base_env.phase == "SCHEDULE":
                print(f" [Cycle {info['hls_state'].current_cycle}] CLOCK TICK | Action: WAIT       | Reward: {reward:+.2f}")
            else:
                print(f" [Transform Phase]   | COMMITTING PRAGMAS AND ENTERING SCHEDULE STAGE | Reward: {reward:+.2f}")
        else:
            node_id = action_val // base_env.PRAGMAS
            pragma = action_val % base_env.PRAGMAS
            
            if base_env.phase == "TRANSFORM":
                pragma_name = ["SCHEDULE", "UNROLL", "PIPELINE", "ARRAY_PART", "DATAFLOW", "INTERFACE"][pragma]
                print(f" [Transform Phase]   | Action: Apply {pragma_name} to Node {node_id:03d} | Area Cost: {info.get('area', 0) - prev_area:+.1f} | Reward: {reward:+.2f}")
            else:
                node = info['hls_state'].nodes[node_id]
                print(f" [Cycle {info['hls_state'].current_cycle}] SCHEDULING | Action: Assign Node {node_id:03d} ({node.op_type[:3]}) to {node.datapath_component} | Reward: {reward:+.2f}")

    cycles = info['hls_state'].current_cycle
    score = grader_func(extract_global_state(obs, info))
    
    print("-" * 80)
    print(f"🏆 {task_id.upper()} COMPLETE!")
    print(f"   Final MakeSpan:     {cycles} Clock Cycles")
    print(f"   Cumulative Reward:  {total_reward:+.2f} (Feedback logic)")
    print(f"   OpenEnv Grader:     {score * 100:.2f}% Pass Rate")
    
    try:
        plot_multi_gantt_chart(
            base_env.execution_history, 
            env, 
            filename=f"gantt_{task_id.lower()}.png",
            title_str=f"HLS Schedule Gantt ({task_id.replace('_', ' ')})"
        )
    except Exception as e:
        print(f"   [Gantt Error]: {e}")
        
    print("\n" * 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_file", type=str, nargs='?', default="kernel.py")
    args = parser.parse_args()
    
    try:
        with open(args.kernel_file, "r") as f:
            source = f.read()
    except Exception as e:
        print(f"Failed to load kernel: {e}")
        exit(1)
        
    print(f"Loaded Native AST Python Benchmark from: {args.kernel_file}\n")
    
    # Task 1: Unconstrained Topology (Infinite Hardware)
    run_evaluation(
        task_id="Task_1_Topology",
        description="Extract the mathematical graph critical-path logic",
        env_params={"max_alu": 999, "max_mac": 999, "max_mem": 999},
        model_file="phase1_unlimited_best_model",
        grader_func=grade_task_1,
        source_code=source
    )
    
    # Task 2: Resource Allocation (Von Neumann Bottleneck)
    run_evaluation(
        task_id="Task_2_Constrained",
        description="Schedule nodes sequentially within rigid boundaries",
        env_params={"max_alu": 2, "max_mac": 1, "max_mem": 1},
        model_file="phase2_constrained_best_model",
        grader_func=grade_task_2,
        source_code=source
    )
    
    # Task 3: Architect (Pragma Scaling with RL Transformers)
    run_evaluation(
        task_id="Task_3_Architect",
        description="Manipulate array structures via PRAGMAS and balance against silicon Area Limits!",
        env_params={"max_alu": 2, "max_mac": 1, "max_mem": 1},
        model_file="phase1_unlimited_best_model", # since phase3 interrupted, phase1 contains the test hooks we fixed
        grader_func=grade_task_3,
        source_code=source
    )
    
