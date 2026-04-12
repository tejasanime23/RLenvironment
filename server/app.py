from fastapi import FastAPI, HTTPException
import uvicorn
import sys
import os
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

# Add root project path to import custom modules natively
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from gauntlet_trainer import make_gauntlet_env
from openenv_wrapper import HLSObservation, HLSAction
from server.graders import grade_task_1, grade_task_2, grade_task_3, clamp_score

app = FastAPI(title="HLS AI Architect OpenEnv API")

# Initialize global hardware environment using standard constrained settings (Gauntlet Pro Mode)
env = make_gauntlet_env(max_alu=4, max_mac=2, max_mem=2)

model_path = "./models/ultimate_gauntlet_agent.zip"
model = None
if os.path.exists(model_path):
    print(f"[UI] Loading trained agent from {model_path}")
    model = MaskablePPO.load(model_path)

current_obs = None
current_info = None

def format_obs(obs, info):
    """ Helper to map internal numpy arrays to the strictly typed OpenEnv Pydantic models """
    return HLSObservation(
        node_features=obs["node_features"].tolist(),
        adj_matrix=obs["adj_matrix"].tolist(),
        global_state=obs["global_state"].tolist(),
        action_mask=obs["action_mask"].tolist()
    )

@app.post("/reset", response_model=HLSObservation)
def reset_env():
    """ OpenEnv Spec: Resets the environment to initial topological constraints and returns the first observation slice. """
    global current_obs, current_info
    obs, info = env.reset()
    current_obs = obs
    current_info = info
    return format_obs(obs, info)

@app.post("/step", response_model=HLSObservation)
def step_env(action: HLSAction):
    """ OpenEnv Spec: Executes a specific flattened action (Pragma or pure Schedule) and advances the environment. """
    global current_obs, current_info
    if current_obs is None:
        raise HTTPException(status_code=400, detail="Environment must be reset first. Call /reset.")
        
    obs, reward, terminated, truncated, info = env.step(action.action_id)
    current_obs = obs
    current_info = info
    return format_obs(obs, info)

@app.get("/state", response_model=HLSObservation)
def get_state():
    """ OpenEnv Spec: Returns the current observation statically without advancing the clock cycle. """
    if current_obs is None:
        raise HTTPException(status_code=400, detail="Environment must be reset first. Call /reset.")
    return format_obs(current_obs, current_info)

def run_inference(code):
    # 1. Hot-swap the UI kernel
    try:
        env.unwrapped.set_kernel(code)
    except Exception as e:
        return f"Error: {e}", None
        
    # Hardcode seed for chart determinism (User requirement)
    obs, info = env.reset(seed=42)
    done = False
    
    while not done:
        if model:
            # Model inference
            action, _ = model.predict(obs, action_masks=env.unwrapped.action_masks(), deterministic=True)
        else:
            # Fallback to current legal schedule if model not yet trained
            action = np.where(env.unwrapped.action_masks() == 1)[0][0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    # 2. Generate visualization
    from run_all_tasks import plot_multi_gantt_chart
    img_path = "/tmp/gantt_temp.png"
    
    try:
        plot_multi_gantt_chart(
            env.unwrapped.execution_history, 
            env, 
            filename=img_path, 
            title_str="User Kernel Schedule"
        )
    except Exception as e:
        return f"Plotting Error: {e}", None
        
    return info.get("status", "Schedule Complete"), img_path

def run_compliance_check():
    """ Runs the 3 official tasks and returns results for the UI table. """
    results = []
    tasks = [
        ("Task 1: Topology", {"max_alu": 999, "max_mac": 999, "max_mem": 999}, grade_task_1, "benchmarks/gauntlet_vector_add.py"),
        ("Task 2: Constrained", {"max_alu": 2, "max_mac": 1, "max_mem": 1}, grade_task_2, "benchmarks/gauntlet_fft_butterfly.py"),
        ("Task 3: Architect", {"max_alu": 2, "max_mac": 1, "max_mem": 1}, grade_task_3, "benchmarks/gauntlet_sobel_stencil.py")
    ]
    
    for name, params, grader, bench_path in tasks:
        try:
            # Setup temp env for this task check
            from gauntlet_trainer import make_gauntlet_env
            t_env = make_gauntlet_env(**params)
            if os.path.exists(bench_path):
                with open(bench_path, "r") as f:
                    source = f.read()
                t_env.unwrapped.set_kernel(source)
            
            # Check phase requirements
            phase = "SCHEDULE" if "Architect" not in name else "TRANSFORM"
            obs, info = t_env.reset(seed=42, options={"initial_phase": phase})
            
            done = False
            while not done:
                if model:
                    action, _ = model.predict(obs, action_masks=t_env.unwrapped.action_masks(), deterministic=True)
                else:
                    action = np.where(t_env.unwrapped.action_masks() == 1)[0][0]
                obs, reward, term, trunc, info = t_env.step(action)
                done = term or trunc
            
            # Grade
            from run_all_tasks import extract_global_state
            score = grader(extract_global_state(obs, info))
            status = "✅ PASS" if score >= 0.7 else "❌ FAIL"
            results.append([name, f"{score*100:.2f}%", status])
            
        except Exception as e:
            results.append([name, "0.00%", f"ERROR: {str(e)[:15]}..."])
            
    return results

# --- Gradio UI "The Judge Magnet" ---
with gr.Blocks(title="HLS AI Architect - Hardware Scheduler", theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")) as demo:
    gr.Markdown("# 🚀 HLS AI Architect\n### Universal Hardware Scheduler (Nemotron-Ready)")
    
    with gr.Tabs():
        with gr.Tab("Interactive Scheduler"):
            gr.Markdown("### 🛠️ Hardware Kernel Profiler\nUpload code and view the agent's cycle-by-cycle scheduling decisions.")
            with gr.Row():
                with gr.Column(scale=1):
                    code_input = gr.Code(label="Python Hardware Kernel", language="python", 
                                        value="def kernel(A, B, C):\n    for i in range(4):\n        C[i] = A[i] + B[i]")
                    run_btn = gr.Button("⚡ Schedule Kernel", variant="primary")
                
                with gr.Column(scale=2):
                    status_box = gr.Textbox(label="Agent Reasoning (status_string)")
                    gantt_plot = gr.Image(label="Architecture Schedule (Gantt Chart)")
            
            run_btn.click(run_inference, inputs=[code_input], outputs=[status_box, gantt_plot])

        with gr.Tab("Compliance Audit"):
            gr.Markdown("## 📉 OpenEnv Gauntlet Scorecard\nRun the official 3-task validation stress-test for your current agent to verify hackathon compliance.")
            audit_btn = gr.Button("🔍 Run Full Compliance Audit", variant="secondary")
            audit_table = gr.Dataframe(
                headers=["Task ID", "Grader Score", "Status"],
                datatype=["str", "str", "str"],
                col_count=(3, "fixed"),
                label="Compliance Report"
            )
            
            audit_btn.click(run_compliance_check, outputs=audit_table)

    gr.Markdown("---")
    gr.Markdown("Built for the **Meta x Scaler OpenEnv Hackathon**. Agent: nvidia/nemotron-3-8b-super | Env: ultimate_gauntlet_v1")

# Mount Gradio into FastAPI (Required for uvicorn server.app:app)
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    """ Entry point for the OpenEnv validator. """
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
