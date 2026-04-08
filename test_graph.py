from hls_env import HLSSchedulerEnv

def main():
    # Write your test pseudo-assembly code here!
    test_code = """
    x = LOAD
    y = LOAD
    a = ADD x y
    b = MUL a 10
    c = SUB a b
    STORE c
    """
    
    print("Loading custom code into HLS Environment...\n")
    try:
        env = HLSSchedulerEnv(source_code=test_code)
        
        print("=== Generated Nodes (Operations) ===")
        for node_id, data in env.base_dag.nodes(data=True):
            print(f"Node {node_id}: Operation '{data['op']}' (Latency = {data['latency']} cycles)")
            
        print("\n=== Generated Edges (Dependencies) ===")
        for parent, child in env.base_dag.edges():
            print(f"Node {parent} MUST finish before Node {child} can start.")
            
        print(f"\nSuccess! Graph created with {len(env.base_dag.nodes)} total operations.")
        
    except Exception as e:
        print(f"Failed to parse graph! Error: {e}")

if __name__ == "__main__":
    main()
