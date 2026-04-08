import time
import psutil
import os
import subprocess

start_time = time.time()
process = psutil.Process()

# Execute inference script
print("Profiling inference start...")
subprocess.run(["python", "evaluate_schedule.py"], capture_output=True)

end_time = time.time()

mem_info = process.memory_info()
print(f"--- Inference Profiling Results ---")
print(f"Wall Clock Time: {end_time - start_time:.3f} seconds")
print(f"Memory Usage (RSS): {mem_info.rss / 1024 / 1024:.2f} MB")
print(f"Memory Usage (VMS): {mem_info.vms / 1024 / 1024:.2f} MB")
print("-----------------------------------")
