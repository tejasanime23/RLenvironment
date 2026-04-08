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

# --- Gradio UI "The Judge Magnet" ---
with gr.Blocks(title="HLS AI Architect - Hardware Scheduler") as demo:
    gr.Markdown("# 🚀 HLS AI Architect\n### Universal Hardware Scheduler (Nemotron-Ready)")
    
    with gr.Row():
        with gr.Column(scale=1):
            code_input = gr.Code(label="Python Hardware Kernel", language="python", value="def kernel(A, B, C):\n    for i in range(4):\n        C[i] = A[i] + B[i]")
            run_btn = gr.Button("Schedule Kernel", variant="primary")
        
        with gr.Column(scale=2):
            status_box = gr.Textbox(label="Agent Reasoning (status_string)")
            gantt_plot = gr.Image(label="Architecture Schedule (Gantt Chart)")
    
    def run_inference(code):
        # 1. Hot-swap the UI kernel
        try:
            env.unwrapped.set_kernel(code)
        except Exception as e:
            return f"Error: {e}", None
            
        obs, info = env.reset()
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
        
        # 2. Generate visualization (reuse run_all_tasks logic or similar)
        from run_all_tasks import generate_gantt_plot
        fig = generate_gantt_plot(env.unwrapped.execution_history, "User_Submission")
        # Save fig to a temp buffer and return
        import matplotlib.pyplot as plt
        img_path = "/tmp/gantt_temp.png"
        fig.savefig(img_path)
        plt.close(fig)
        
        return info.get("status", "Schedule Complete"), img_path

    run_btn.click(run_inference, inputs=[code_input], outputs=[status_box, gantt_plot])

# Mount Gradio into FastAPI (Required for uvicorn server.app:app)
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
