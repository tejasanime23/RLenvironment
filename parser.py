import networkx as nx

def build_dag_from_code(source_code: str) -> nx.DiGraph:
    """
    A simple parser that reads a sequence of assembly-like operations 
    and builds a Data Flow Graph (DAG) with dependencies.
    
    Format example:
    a = LOAD
    b = ADD a 5
    c = MUL a 10
    d = ADD b c
    STORE d
    """
    dag = nx.DiGraph()
    var_to_node = {} # Maps a variable like 'a' to the Node ID that created it
    
    # Simple hardcoded latencies for our toy hardware
    latencies = {
        "LOAD": 2,
        "STORE": 2,
        "ADD": 1,
        "MUL": 3,
        "SUB": 1
    }
    
    node_id = 0
    lines = source_code.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue # Skip empty lines and comments
            
        parts = line.split()
        
        # Check if this instruction assigns a value to a variable
        # Example: b = ADD a 5
        if "=" in parts:
            eq_idx = parts.index("=")
            dest_var = parts[0]
            op = parts[eq_idx + 1].upper()
            args = parts[eq_idx + 2:]
        else:
            # Example: STORE d
            dest_var = None
            op = parts[0].upper()
            args = parts[1:]
            
        latency = latencies.get(op, 1) # Default latency is 1
        
        # Add the operation as a node to the graph
        dag.add_node(node_id, op=op, latency=latency)
        
        # Draw Data Dependency Edges
        # If any argument was created by a previous instruction, we must wait for it!
        for arg in args:
            if arg in var_to_node:
                parent_id = var_to_node[arg]
                # parent must finish BEFORE current node_id starts
                dag.add_edge(parent_id, node_id)
                
        # Register our output variable so future instructions can depend on us
        if dest_var:
            var_to_node[dest_var] = node_id
            
        node_id += 1
        
    return dag
