import asyncio
import os
import textwrap
import numpy as np
import gymnasium as gym
from typing import List, Optional
from openai import OpenAI

# Import your hardened environment
from hls_env import HLSSchedulerEnv

# --- Configuration & Environment Variables ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/nemotron-3-8b-super")
API_KEY = os.getenv("HF_TOKEN")

# Hardware Task Configuration
# The judge passes TASK_IDs, we map them to our Gauntlet Stressors
TASK_MAPPING = {
    "task_1_topology": "gauntlet_vector_add",
    "task_2_constrained": "gauntlet_fft_butterfly",
    "task_3_architect": "gauntlet_sobel_stencil"
}

RAW_TASK = os.getenv("HLS_TASK", "task_2_constrained")
TASK_NAME = TASK_MAPPING.get(RAW_TASK, RAW_TASK) 
BENCHMARK = "ultimate_gauntlet_v1"
MAX_STEPS = 500 

# Resource limits as defined in our Phase 2/3 mastery
RESOURCES = {"ALU": 4, "MAC": 2, "MEM": 2}

# --- The "Translator" (Prompt Engineering for Nemotron) ---
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Hardware Architect. Your goal is to schedule a Data Dependency Graph (DAG) 
    onto a constrained silicon architecture: 4 ALUs, 2 MAC units, and 2 Memory Ports.
    
    RULES:
    1. Minimize Latency: Schedule nodes as soon as their dependencies are met and resources are free.
    2. Respect Dependencies: You cannot schedule a node if its parents haven't finished.
    3. Resource Awareness: If all ports are busy, you MUST use the "WAIT" command.
    4. Format: Respond with ONLY the action string: "SCHEDULE <NodeID>" or "WAIT".
    """
).strip()

def build_user_prompt(obs, status_str, history: List[str]) -> str:
    # Feature 5 is our "Ready Node Count" (from hls_env.py _get_obs global_state)
    ready_count = int(obs[4])
    history_block = "\n".join(history[-3:]) if history else "Beginning of execution."
    
    return textwrap.dedent(
        f"""
        CURRENT HARDWARE STATE:
        {status_str}
        Nodes ready for scheduling: {ready_count}
        
        RECENT ACTIONS:
        {history_block}
        
        Next action? (Format: SCHEDULE <ID> or WAIT)
        """
    ).strip()

# --- STDOUT Logging (Required for Hackathon Grading) ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

class OpenEnvStringWrapper(gym.Wrapper):
    """
    Compatibility layer that translates LLM string actions to HLSSchedulerEnv discrete actions.
    """
    def step(self, action_str: str):
        # 1. Parse "SCHEDULE <ID>" or "WAIT"
        action_idx = self.env.unwrapped.BUTTON_ACTION # Default to WAIT
        
        try:
            action_str = action_str.strip().upper()
            if action_str.startswith("SCHEDULE"):
                node_id = int(action_str.split()[-1])
                # We assume Phase 2 (SCHEDULE) and Pragma 0
                action_idx = node_id * self.env.unwrapped.PRAGMAS
            elif "WAIT" in action_str:
                action_idx = self.env.unwrapped.BUTTON_ACTION
        except Exception:
            pass # Default to WAIT if malformed
            
        obs, reward, term, trunc, info = self.env.step(action_idx)
        return obs["global_state"], reward, (term or trunc), info

    def reset(self, kernel_name=None):
        # Hot-swap to the kernel file
        if kernel_name:
            kernel_path = f"benchmarks/{kernel_name}.py"
            if os.path.exists(kernel_path):
                with open(kernel_path, "r") as f:
                    source = f.read()
                self.env.unwrapped.set_kernel(source)
            
        obs, info = self.env.reset(options={"initial_phase": "SCHEDULE"})
        return obs["global_state"], info

async def eval_loop():
    # Detect the HF environment or fail gracefully
    if not API_KEY:
        print("[ENV ERROR] HF_TOKEN is missing. Simulation will use 'WAIT' for all steps.")
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    
    # Initialize Environment
    raw_env = HLSSchedulerEnv(
        max_alu=RESOURCES["ALU"],
        max_mac=RESOURCES["MAC"],
        max_mem=RESOURCES["MEM"]
    )
    env = OpenEnvStringWrapper(raw_env)
    
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Step into scheduled mode immediately for evaluation
        obs, info = env.reset(kernel_name=TASK_NAME)
        status_str = info.get("status", "System initialized.")
        
        for step in range(1, MAX_STEPS + 1):
            if client:
                prompt = build_user_prompt(obs, status_str, history)
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=20
                )
                action_str = (completion.choices[0].message.content or "WAIT").strip()
            else:
                action_str = "WAIT" # Fallback for local testing without API key

            # Step the environment
            obs, reward, done, info = env.step(action_str)
            
            status_str = info.get("status", "No status updates.")
            error = info.get("synthesis_error", None) # Matches our hardened hls_env.py
            rewards.append(reward)
            steps_taken = step
            
            log_step(step, action_str, reward, done, error)
            history.append(f"Cycle {step}: {action_str} -> {status_str}")

            if done:
                break

        # Calculate final score (normalized based on critical path depth)
        meta = info.get("metadata", {})
        cp = meta.get("critical_path_depth", 1)
        # Simple score based on makespan vs critical path
        final_score = max(0.0, min(1.0, cp / max(1, info["hls_state"].current_cycle)))
        success = final_score >= 0.7 

    except Exception as e:
        print(f"[ERROR] Inference crashed: {e}")
    finally:
        log_end(success, steps_taken, final_score, rewards)

if __name__ == "__main__":
    asyncio.run(eval_loop())
