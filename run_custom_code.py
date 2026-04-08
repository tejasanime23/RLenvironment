import sys
import argparse
import networkx as nx
from hls_env import HLSSchedulerEnv

# Try importing matplotlib, but don't crash if it's missing
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def show_graph(dag):
    if not MATPLOTLIB_AVAILABLE:
        print("\n[Notice] Matplotlib is not installed. Run `pip install matplotlib` to see the 2D Graph Visualizer!")
        return
        
    print("\nRendering 2D Hardware DAG Window...")
    plt.figure(figsize=(10, 8))
    
    # Topological sort for hierarchical left-to-right logic flow layout
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(dag, prog="dot")
    except Exception as e:
        # Fallback to standard spring layout if Graphviz executable 'dot' isn't installed
        print("    (Note: Graphviz 'dot' not installed in PATH, falling back to spring_layout)")
        pos = nx.spring_layout(dag, seed=42)
        
    # Color code nodes based on Datapath Component!
    color_map = []
    labels = {}
    for node_id, data in dag.nodes(data=True):
        component = data.get('datapath_component', 'NONE')
        if component == "ALU":
            color_map.append("lightblue")
        elif component == "MAC":
            color_map.append("lightcoral")
        elif component == "MEM":
            color_map.append("lightgreen")
        else:
            color_map.append("lightgrey")
            
        labels[node_id] = f"{node_id}\n{data.get('op')}"

    nx.draw(dag, pos, node_color=color_map, with_labels=True, labels=labels, 
            node_size=1500, font_size=9, font_weight="bold", arrows=True,
            edge_color="gray", width=1.5)
            
    plt.title("HLS Unrolled Data Dependency Graph\n(Blue=ALU, Red=MAC, Green=MEM)", fontsize=14)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run HLS Scheduler on a custom python kernel file.")
    parser.add_argument("kernel_file", type=str, nargs='?', default=None, help="Path to your python kernel file (Optional).")
    args = parser.parse_args()
    
    if args.kernel_file:
        try:
            with open(args.kernel_file, "r") as f:
                source_code = f.read()
            print(f"Compiling '{args.kernel_file}' and allocating Datapath Components...")
        except Exception as e:
            print(f"Error reading {args.kernel_file}: {e}")
            sys.exit(1)
    else:
        print("=== Interactive HLS Compilation Mode ===")
        print("Type or paste your Python code below.")
        print("When you are finished, press ENTER twice on an empty line to compile:")
        print("-" * 40)
        lines = []
        while True:
            try:
                line = input()
                if line == "" and (len(lines) > 0 and lines[-1] == ""):
                    break
                lines.append(line)
            except EOFError:
                break
        source_code = "\n".join(lines)
        print("-" * 40)
        print("Compiling Interactive Code and allocating Datapath Components...")
    
    try:
        env = HLSSchedulerEnv(source_code=source_code)
        
        # We must call reset() to fully initialize the inner state parameters like registers and hardware
        env.reset()
        
        print("\n=== Datapath Component & Register Allocation ===")
        for node_id, data in env.base_dag.nodes(data=True):
            print(f"Node {node_id} ({data.get('desc')}):")
            print(f"  -> Operation: {data.get('op')}")
            print(f"  -> Datapath Required: {data.get('datapath_component')} (Latency = {data.get('latency')} cycles)")
            print(f"  -> Target Register: {data.get('target_register')}")
            print()
            
        print("=== Generated Hardware Dependencies ===")
        for parent, child in env.base_dag.edges():
            print(f"Node {parent} MUST finish before Node {child} can start.")
            
        # Display Hardware Limits
        hw = env.hls_state.hardware
        print(f"\n=== Environment Hardware Constraints ===")
        print(f"ALUs Available: {hw.total_alu}")
        print(f"MACs (Multipliers) Available: {hw.total_mac}")
        print(f"MEM Interfaces Available: {hw.total_mem}")
        
        # Pop up the visualizer
        show_graph(env.base_dag)
            
    except Exception as e:
        print(f"Failed to compile Python! Error: {e}")

if __name__ == "__main__":
    main()
