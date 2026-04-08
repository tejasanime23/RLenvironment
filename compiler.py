import ast
import inspect
import networkx as nx

class HLSSynthesisError(Exception):
    """ Custom exception triggered when Python logic violates static hardware synthesis rules. """
    pass


def generate_vector_add_dag(loop_iterations=4):
    """
    Generates a mock DAG representing the bottleneck vector loop:
    for i in range(N): C[i] = A[i] + B[i]
    """
    dag = nx.DiGraph()
    node_id = 0
    
    last_iter_store = None
    
    for i in range(loop_iterations):
        load_a = node_id; node_id += 1
        load_b = node_id; node_id += 1
        add_op = node_id; node_id += 1
        store_c = node_id; node_id += 1
        
        dag.add_node(load_a, op="LOAD", latency=3, datapath_component="MEM", target_register=f"A_{i}")
        dag.add_node(load_b, op="LOAD", latency=3, datapath_component="MEM", target_register=f"B_{i}")
        dag.add_node(add_op, op="ADD", latency=1, datapath_component="ALU", target_register=f"C_{i}_val")
        dag.add_node(store_c, op="STORE", latency=2, datapath_component="MEM", target_register=f"C_{i}")
        
        # Intra-iteration data dependencies
        dag.add_edge(load_a, add_op)
        dag.add_edge(load_b, add_op)
        dag.add_edge(add_op, store_c)
        
        # Inter-iteration dependencies to represent standard sequential loop processing
        if last_iter_store is not None:
            dag.add_edge(last_iter_store, load_a, is_loop_carried=True)
            dag.add_edge(last_iter_store, load_b, is_loop_carried=True)
            
        last_iter_store = store_c
        
    # Add a global sink to represent function exit
    sink_id = node_id
    dag.add_node(sink_id, op="SINK", latency=0, datapath_component="ALU", target_register="SINK", desc="GLOBAL SINK")
    dag.add_edge(last_iter_store, sink_id)
        
    return dag

class HLSCompilerWalker(ast.NodeVisitor):
    def __init__(self):
        self.dag = nx.DiGraph()
        self.node_id = 0
        self.variables = {}
        self.register_counter = 0
        
        # Track side-effects to create sequential Control Dependencies
        self.last_side_effect_node = None
        
        self.hardware_map = {
            "LOAD": {"latency": 2, "component": "MEM"},
            "STORE": {"latency": 2, "component": "MEM"},
            "ADD": {"latency": 1, "component": "ALU"},
            "SUB": {"latency": 1, "component": "ALU"},
            "MUL": {"latency": 3, "component": "MAC"},
            "DIV": {"latency": 4, "component": "ALU"},
            "CALL": {"latency": 2, "component": "ALU"},
            "BRANCH": {"latency": 1, "component": "ALU"}
        }

    def _add_node(self, op, desc, deps, is_side_effect=False):
        if self.node_id > 1000:
            raise Exception("HLS Compiler Limit Reached: Generated graph exceeded 1000 nodes!")

        hw = self.hardware_map.get(op, {"latency": 1, "component": "ALU"})
        target_reg = f"R{self.register_counter}"
        self.register_counter += 1
        
        nid = self.node_id
        self.dag.add_node(nid, op=op, latency=hw["latency"], 
                          datapath_component=hw["component"], 
                          target_register=target_reg, desc=desc)
        
        # Add actual Data Dependencies
        for dep in deps:
            self.dag.add_edge(dep, nid)
            
        # Handle Control Dependencies (Sequential chaining for I/O / branches)
        if is_side_effect:
            if self.last_side_effect_node is not None:
                # Force this node to wait for the previous side-effect to finish!
                if self.last_side_effect_node != nid:
                    self.dag.add_edge(self.last_side_effect_node, nid)
            self.last_side_effect_node = nid
            
        self.node_id += 1
        return nid

    def visit_Assign(self, node):
        value_node_id = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables[target.id] = value_node_id
            elif isinstance(target, ast.Subscript):
                # Hardware Store Operation (Memory Write)
                arr_id = self.visit(target.value)
                idx_id = self.visit(target.slice)
                self._add_node("STORE", "ARRAY STORE", [arr_id, idx_id, value_node_id])

    def visit_AugAssign(self, node):
        # x += y -> x = x + y
        right_id = self.visit(node.value)
        
        # Handle the "Load" of the target first
        if isinstance(node.target, ast.Name):
            left_id = self.variables.get(node.target.id)
            if left_id is None:
                left_id = self._add_node("LOAD", f"ARG '{node.target.id}'", [])
        elif isinstance(node.target, ast.Subscript):
            left_id = self.visit(node.target)
        else:
            return

        op_type = type(node.op)
        if op_type == ast.Add: op = "ADD"
        elif op_type == ast.Sub: op = "SUB"
        elif op_type == ast.Mult: op = "MUL"
        elif op_type == ast.Div: op = "DIV"
        else: op = "ADD"

        result_id = self._add_node(op, op, [left_id, right_id])
        
        # Store result back
        if isinstance(node.target, ast.Name):
            self.variables[node.target.id] = result_id
        elif isinstance(node.target, ast.Subscript):
            arr_id = self.visit(node.target.value)
            idx_id = self.visit(node.target.slice)
            self._add_node("STORE", "ARRAY STORE", [arr_id, idx_id, result_id])

    def visit_BinOp(self, node):
        left_id = self.visit(node.left)
        right_id = self.visit(node.right)
        
        op_type = type(node.op)
        if op_type == ast.Add: op = "ADD"
        elif op_type == ast.Sub: op = "SUB"
        elif op_type == ast.Mult: op = "MUL"
        elif op_type == ast.Div: op = "DIV"
        else: op = "ADD" 
        
        return self._add_node(op, op, [left_id, right_id])

    def visit_Compare(self, node):
        # We handle comparisons (e.g., choice == 1) same as BinOp for Data Flow ALUs
        left_id = self.visit(node.left)
        
        # NOTE: Python allows chained comparisons (a < b < c). We'll simplify to just the first one.
        right_id = self.visit(node.comparators[0])
        
        return self._add_node("ALU", type(node.ops[0]).__name__.upper(), [left_id, right_id])

    def visit_UnaryOp(self, node):
        operand_id = self.visit(node.operand)
        op_name = type(node.op).__name__.upper()
        # In HLS, a negation is often just a Subtraction from zero or a specific ALU primitive
        return self._add_node("ALU", f"UNARY_{op_name}", [operand_id])

    def visit_BoolOp(self, node):
        # logic for 'and' / 'or'
        val_ids = [self.visit(val) for val in node.values]
        op_name = type(node.op).__name__.upper()
        return self._add_node("ALU", f"BOOL_{op_name}", val_ids)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id in self.variables:
                return self.variables[node.id]
            else:
                return self._add_node("LOAD", f"ARG '{node.id}'", [])

    def visit_Constant(self, node):
        return self._add_node("LOAD", f"CONST {node.value}", [])

    def visit_Call(self, node):
        # We need to process arguments first (Data dependencies)
        arg_ids = []
        for arg in node.args:
            arg_ids.append(self.visit(arg))
            
        func_name = "func"
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            # We treat the object being called on as a data dependency
            obj_id = self.visit(node.func.value)
            arg_ids.append(obj_id)

        # Function calls inherently have side effects (like print, input, pop, append)
        # So we flag them as is_side_effect=True to chain them chronologically
        return self._add_node("CALL", f"CALL '{func_name}'", arg_ids, is_side_effect=True)
        
    def visit_Subscript(self, node):
        # Hardware Load Operation (Memory Read)
        arr_id = self.visit(node.value)
        idx_id = self.visit(node.slice)
        return self._add_node("LOAD", "ARRAY LOAD", [arr_id, idx_id])
        
    def visit_Expr(self, node):
        # Top level expressions (like a standalone print statement)
        self.visit(node.value)

    def visit_If(self, node):
        # Branching Logic!
        # The boolean check itself is a data root
        test_id = self.visit(node.test)
        
        # We mark the branching decision as a Side-Effect to force sequence!
        branch_id = self._add_node("BRANCH", "IF Branch MUX", [test_id], is_side_effect=True)
        
        # Then visit inside the block
        for stmt in node.body:
            self.visit(stmt)
            
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and getattr(node.iter.func, 'id', '') == "range":
            args = node.iter.args
            if len(args) == 1:
                start = 0
                end = args[0].value
            else:
                start = args[0].value
                end = args[1].value
                
            iter_var = node.target.id
            
            for i in range(start, end):
                iter_node_id = self._add_node("LOAD", f"ITER {i}", [])
                self.variables[iter_var] = iter_node_id
                for stmt in node.body:
                    self.visit(stmt)
        else:
            raise Exception("Compiler currently only supports 'range()' loops for unrolling.")

    def visit_Return(self, node):
        if node.value:
            if isinstance(node.value, ast.Tuple):
                ret_ids = [self.visit(elt) for elt in node.value.elts]
                self._add_node("STORE", "RETURN", ret_ids, is_side_effect=True)
            else:
                ret_id = self.visit(node.value)
                self._add_node("STORE", "RETURN", [ret_id], is_side_effect=True)
            
    # --- UNSUPPORTED DYNAMIC HARDWARE TRAPS ---
    def _reject_dynamic(self, node):
        raise HLSSynthesisError(f"HLS Compiler Error: Dynamic Python functionality '{node.__class__.__name__}' cannot be synthesized into static hardware! Only statically bounded operations are permitted.")

    def visit_While(self, node): self._reject_dynamic(node)
    def visit_DictComp(self, node): self._reject_dynamic(node)
    def visit_ListComp(self, node): self._reject_dynamic(node)
    def visit_Dict(self, node): self._reject_dynamic(node)
    def visit_List(self, node): self._reject_dynamic(node)
    def visit_Import(self, node): self._reject_dynamic(node)
    def visit_ImportFrom(self, node): self._reject_dynamic(node)


def compile_python_to_dag(source_code) -> nx.DiGraph:
    if not isinstance(source_code, str):
        source_code = inspect.getsource(source_code)
        
    tree = ast.parse(source_code)
    
    compiler = HLSCompilerWalker()
    compiler.visit(tree)
    
    dag = compiler.dag
    
    # --- POST-PROCESSING: Global Target Sink Node (Node 48 equivalent) ---
    # Find all "Leaf Nodes" (nodes that have no outgoing connections)
    # We sort to ensure determinism in edge creation order
    leaf_nodes = sorted([n for n in dag.nodes() if dag.out_degree(n) == 0])
    
    # Don't bother inserting sink if graph is empty
    if len(leaf_nodes) > 0:
        sink_id = compiler.node_id
        dag.add_node(sink_id, op="SINK", latency=0, datapath_component="ALU", 
                     target_register="SINK", desc="GLOBAL SINK / TERMINATE")
                     
        # String them all up to the great Sink!
        for leaf in leaf_nodes:
            dag.add_edge(leaf, sink_id)
    
    return dag
