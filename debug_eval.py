import os
from sb3_contrib import MaskablePPO
from curriculum_trainer import make_env

model_path = os.path.join('models', 'phase1_unlimited_best_model')
model = MaskablePPO.load(model_path)
env = make_env(max_alu=2, max_mac=1, max_mem=1)
obs, info = env.reset()

print('Started')
for i in range(20):
    action, _states = model.predict(obs, action_masks=obs['action_mask'], deterministic=True)
    obs, r, term, trunc, info = env.step(int(action))
    print(f"Step {i}: Action {action}, Phase {env.unwrapped.phase}, Cycle {info['hls_state'].current_cycle if 'hls_state' in info else 0}")
    if term or trunc: 
        print("Done!")
        break
