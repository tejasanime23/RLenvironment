from hls_env import HLSSchedulerEnv

def my_hardware_kernel(a, b):
    # This is REAL standard Python code!
    x = a + 5
    y = x * b
    return y

def main():
    print("Loading native Python function into HLS Environment...\n")
    try:
        env = HLSSchedulerEnv(python_function=my_hardware_kernel)
        
        print("=== Native Python Bytecode Compiled to Operations ===")
        for node_id, data in env.base_dag.nodes(data=True):
            print(f"Node {node_id}: {data['desc']} -> Assigned Operation '{data['op']}' (Latency = {data['latency']} cycles)")
            
        print("\n=== Automatically Inferred Hardware Dependencies ===")
        for parent, child in env.base_dag.edges():
            print(f"Node {parent} MUST finish before Node {child} can start.")
            
        print(f"\nSuccess! Graph created automatically from Python code using Symbolic Execution!")
        
    except Exception as e:
        print(f"Failed to compile Python! Error: {e}")

if __name__ == "__main__":
    main()
