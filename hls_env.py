import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
from models import NodeState, HLSState, State
from compiler import compile_python_to_dag, generate_vector_add_dag

OPCODE_MAP = {
    "UNKNOWN": 0,
    "LOAD": 1,
    "STORE": 2,
    "ADD": 3,
    "SUB": 4,
    "MUL": 5,
    "DIV": 6,
    "CALL": 7,
    "BRANCH": 8,
    "SINK": 9,
}

class HLSSchedulerEnv(gym.Env):
    """
    Two-Phase Gymnasium environment for HLS Hardware Scheduling.
    Phase 1: TRANSFORM (Applies 5 Pragmas, mutates DAG, increases Area)
    Phase 2: SCHEDULE  (Cycle-by-cycle scheduling with Pareto boundaries)
    """
    
    def __init__(self, source_code=None, python_function=None, max_alu=4, max_mac=2, max_mem=2, universal_max_nodes=300, max_area=1000.0):
        super(HLSSchedulerEnv, self).__init__()
        
        self.max_alu = max_alu
        self.max_mac = max_mac
        self.max_mem = max_mem
        
        self.synthesis_error = False
        try:
            if source_code is not None:
                self.base_dag = compile_python_to_dag(source_code)
            else:
                self.base_dag = generate_vector_add_dag(loop_iterations=4)
        except Exception as e:
            # Catch synthesis errors to prevent environment crashes during gauntlet runs
            print(f"[ENV WARNING]: Synthesis failed: {e}")
            self.synthesis_error = True
            # Fallback to a single-node dummy graph so the observation space doesn't break
            self.base_dag = nx.DiGraph()
            self.base_dag.add_node(0, op="SINK", latency=0, datapath_component="ALU", target_register="SINK")
        
        self.dag = None
        self.hls_state = None
        self.node_criticality = {}
        
        self.real_nodes = len(self.base_dag.nodes)
        
        if self.real_nodes > universal_max_nodes:
            raise Exception(f"Input graph has {self.real_nodes} nodes, which exceeds universal_max_nodes.")
            
        self.max_nodes = universal_max_nodes
        
        # Flattened Constraints: 0=Schedule, 1=Unroll, 2=Pipeline, 3=Partition, 4=Dataflow, 5=Interface
        self.PRAGMAS = 6
        
        # Action self.max_nodes * 6 is the dedicated phase switch / WAIT button
        self.BUTTON_ACTION = self.max_nodes * self.PRAGMAS
        
        # Action space: Discrete(MAX_NODES * 6 + 1)
        self.action_space = spaces.Discrete(self.BUTTON_ACTION + 1)
        
        # Observation space: 
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(low=-1.0, high=1000.0, shape=(self.max_nodes, 6), dtype=np.float32),
            "adj_matrix": spaces.Box(low=0.0, high=1.0, shape=(self.max_nodes, self.max_nodes), dtype=np.float32),
            "global_state": spaces.Box(low=0.0, high=1000.0, shape=(5,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.BUTTON_ACTION + 1,), dtype=np.int8)
        })
        
        # Pareto Limits
        self.BASE_AREA_PER_NODE = 10.0
        # Automatically set constraint ceiling based on initial graph baseline bounds!
        self.MAX_AREA = max(1000.0, float(self.max_nodes * 20.0))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Default to TRANSFORM, but allow skipping for pure scheduling tasks
        self.phase = options.get("initial_phase", "TRANSFORM") if options else "TRANSFORM"
        self.current_area = 0.0
        self.execution_history = []
        self.dag = self.base_dag.copy()
        
        self.node_pragmas = {n: 0 for n in list(self.dag.nodes())}
        self.real_nodes = len(self.dag.nodes())
        
        for n in self.dag.nodes():
            self.current_area += self.BASE_AREA_PER_NODE
            
        self._recompute_criticality()
        self._init_hls_state()
        
        # Calculate optimal baselines once for the grader metadata
        if not hasattr(self, 'baseline_critical_path'):
            try:
                self.baseline_critical_path = nx.dag_longest_path_length(self.base_dag)
            except Exception:
                self.baseline_critical_path = 0
            self.baseline_total_ops = len(self.base_dag.nodes)
        
        return self._get_obs(), self._get_info()

    def set_kernel(self, source_code=None):
        """
        Hot-swaps the underlying hardware kernel.
        """
        self.synthesis_error = False
        try:
            if source_code is not None:
                self.base_dag = compile_python_to_dag(source_code)
            else:
                self.base_dag = generate_vector_add_dag(loop_iterations=4)
        except Exception as e:
            print(f"[ENV WARNING]: Synthesis failed during hot-swap: {e}")
            self.synthesis_error = True
            self.base_dag = nx.DiGraph()
            self.base_dag.add_node(0, op="SINK", latency=0, datapath_component="ALU", target_register="SINK")
            
        self.real_nodes = len(self.base_dag.nodes)
        if self.real_nodes > self.max_nodes:
            # If a gauntlet kernel is too large, we fall back to a smaller version or raise an error
            raise ValueError(f"Gauntlet kernel size {self.real_nodes} exceeds environment limit {self.max_nodes}")
            
        # Clear specific baseline caches
        if hasattr(self, 'baseline_critical_path'):
            delattr(self, 'baseline_critical_path')
        
        # Reset internal logic for the new graph
        self.reset()
        
    def _recompute_criticality(self):
        self.node_criticality = {}
        for node in sorted(list(self.dag.nodes())):
            try:
                self.node_criticality[node] = len(nx.descendants(self.dag, node))
            except nx.NetworkXError:
                self.node_criticality[node] = 0

    def _init_hls_state(self):
        nodes = {}
        # Ensure nodes are initialized in deterministic ID order
        for node_id, data in sorted(self.dag.nodes(data=True)):
            nodes[node_id] = NodeState(
                node_id=node_id,
                op_type=data.get("op", "UNKNOWN"),
                latency=data.get("latency", 1),
                datapath_component=data.get("datapath_component", "NONE"),
                target_register=data.get("target_register", ""),
                state=State.PENDING,
                cycles_remaining=0
            )
            
        # Retain references securely if already mid-scheduling
        curr_cyc = self.hls_state.current_cycle if self.hls_state else 0
        self.hls_state = HLSState(current_cycle=curr_cyc, nodes=nodes)
        self.hls_state.hardware.total_alu = self.max_alu
        self.hls_state.hardware.total_mac = self.max_mac
        self.hls_state.hardware.total_mem = self.max_mem
        
    def _apply_pragma(self, node_id, pragma):
        # Prevent re-applying pragmas to the same node to avoid infinite loops
        if self.node_pragmas.get(node_id, 0) > 0:
            return False
            
        self.node_pragmas[node_id] = pragma
        
        if pragma == 1: # UNROLL
            # Duplicate the node physically to allow perfect parallel compilation
            new_id = self.real_nodes
            self.real_nodes += 1
            if self.real_nodes > self.max_nodes:
                return False
                
            data = self.dag.nodes[node_id]
            self.dag.add_node(new_id, **data)
            
            # Map parent edges to duplicate
            for p in list(self.dag.predecessors(node_id)):
                edge_data = self.dag.get_edge_data(p, node_id)
                self.dag.add_edge(p, new_id, **edge_data)
                
            # Map child edges from duplicate
            for c in list(self.dag.successors(node_id)):
                edge_data = self.dag.get_edge_data(node_id, c)
                self.dag.add_edge(new_id, c, **edge_data)
                
            self.current_area += self.BASE_AREA_PER_NODE
            self.node_pragmas[new_id] = pragma
            
        elif pragma == 2: # PIPELINE
            # Sever loop-carried dependencies to allow temporal overlap
            edges_to_remove = []
            for u, v, data in self.dag.edges(data=True):
                if data.get('is_loop_carried', False) and (u == node_id or v == node_id):
                    edges_to_remove.append((u, v))
            for u, v in edges_to_remove:
                self.dag.remove_edge(u, v)
            self.current_area += 15.0 # Register area cost
            
        elif pragma == 3: # ARRAY_PARTITION
            # Split memories to double throughput
            if self.dag.nodes[node_id].get("datapath_component") == "MEM":
                self.max_mem += 1
                self.current_area += 50.0 # Heavy memory controller cost
                
        elif pragma == 4: # DATAFLOW
            # Drop hard dependencies into streaming FIFOs (0 latency boundaries)
            for c in list(self.dag.successors(node_id)):
                self.dag.remove_edge(node_id, c)
            self.current_area += 30.0 # FIFO cost
            
        elif pragma == 5: # INTERFACE
            self.dag.nodes[node_id]["latency"] = 1
            self.current_area += 5.0 # AXI Stream wrapping cost
            
        return True
        
    def _get_obs(self):
        node_features = np.zeros((self.max_nodes, 6), dtype=np.float32)
        adj_matrix = np.zeros((self.max_nodes, self.max_nodes), dtype=np.float32)
        # Initialize mask: [node_0_pragma_0...node_N_pragma_5, WAIT_BUTTON]
        mask_size = self.BUTTON_ACTION + 1
        action_mask = np.zeros(mask_size, dtype=np.int8)
        
        if self.real_nodes > 0:
            nodelist = sorted(list(self.dag.nodes()))
            real_adj = nx.to_numpy_array(self.dag, nodelist=nodelist)
            adj_matrix[:self.real_nodes, :self.real_nodes] = real_adj
            
        action_mask[self.BUTTON_ACTION] = 1 # Global WAIT button is always valid
        
        nodes_completed = 0
        total_nodes = self.real_nodes if self.real_nodes > 0 else 1
        
        for i in range(self.real_nodes):
            if i in self.hls_state.nodes:
                node = self.hls_state.nodes[i]
                # In-Degree (unfinished parents)
                unfinished_parents = 0
                for p in self.dag.predecessors(i):
                    # Safely handle dynamically spawned UNROLL nodes that haven't been committed to hls_state yet
                    n_state = self.hls_state.nodes.get(p)
                    if n_state is None or n_state.state != State.COMPLETED:
                        unfinished_parents += 1
                
                op_code = OPCODE_MAP.get(node.op_type.upper(), 0)
                pragma_state = float(self.node_pragmas.get(i, 0))
                
                node_features[i, 0] = op_code
                node_features[i, 1] = float(node.state.value)
                node_features[i, 2] = float(node.latency)
                node_features[i, 3] = float(unfinished_parents)
                node_features[i, 4] = float(self.node_criticality.get(i, 0))
                node_features[i, 5] = pragma_state
                
                if node.state == State.COMPLETED:
                    nodes_completed += 1
                
                # --- TWO-PHASE MASKING LOGIC ---
                if self.phase == "TRANSFORM":
                    # Mask out Action 0 (Schedule)
                    if pragma_state == 0:
                        idx_base = i * self.PRAGMAS
                        action_mask[idx_base + 1] = 1 # Unroll
                        action_mask[idx_base + 2] = 1 # Pipeline
                        if node.datapath_component == "MEM":
                            action_mask[idx_base + 3] = 1 # Array Partition
                        action_mask[idx_base + 4] = 1 # Dataflow
                        if node.op_type in ["LOAD", "STORE"]:
                            action_mask[idx_base + 5] = 1 # Interface
                        
                elif self.phase == "SCHEDULE":
                    # Mask out all Pragma modifications
                    if self._is_valid_schedule_action(i):
                        idx_base = i * self.PRAGMAS
                        action_mask[idx_base + 0] = 1 # The Schedule execution
                        
        # Enforce Greedy Scheduling: Prevent infinite WAIT loops by masking WAIT if parallel paths are fully available
        if self.phase == "SCHEDULE":
            # Check if any schedule action is 1
            has_valid_action = False
            for i in range(self.real_nodes):
                if action_mask[i * self.PRAGMAS] == 1:
                    has_valid_action = True
                    break
            
            # If we CAN schedule something, disable the WAIT button!
            if has_valid_action:
                action_mask[self.BUTTON_ACTION] = 0
                    
        progress_ratio = nodes_completed / total_nodes
        active_alu = self.hls_state.hardware.active_alu
        active_mac = self.hls_state.hardware.active_mac
        active_mem = self.hls_state.hardware.active_mem
        
        # Calculate Ready Nodes (Dynamic frontier sensing for Nemotron)
        ready_nodes = 0
        if self.phase == "SCHEDULE":
            for i in range(self.real_nodes):
                mask_idx = i * self.PRAGMAS
                if mask_idx < self.BUTTON_ACTION and action_mask[mask_idx] == 1:
                    ready_nodes += 1
        
        global_state = np.array([
            float(self.hls_state.current_cycle),
            float(active_alu + active_mac + active_mem), # Total busy resources
            float(progress_ratio),
            float(self.current_area / self.MAX_AREA), # Area used ratio
            float(ready_nodes) # New Feature #5
        ], dtype=np.float32)

        return {
            "node_features": node_features,
            "adj_matrix": adj_matrix,
            "global_state": global_state,
            "action_mask": action_mask
        }
        
    def action_masks(self):
        return self._get_obs()["action_mask"]

    def _get_info(self):
        # Build human-readable status for Nemotron's <think> block
        hw = self.hls_state.hardware
        active_res = hw.active_alu + hw.active_mac + hw.active_mem
        total_res = hw.total_alu + hw.total_mac + hw.total_mem
        
        status_str = f"Phase: {self.phase} | Cycle: {self.hls_state.current_cycle} | "
        status_str += f"Resources: {active_res}/{total_res} busy | "
        
        if self.phase == "SCHEDULE":
            # Count ready but not executing
            ready = sum(1 for i in range(self.real_nodes) if self._is_valid_schedule_action(i))
            status_str += f"{ready} nodes ready to schedule."
        else:
            status_str += f"Building hardware architecture..."

        return {
            "hls_state": self.hls_state, 
            "area": self.current_area,
            "status": status_str,
            "synthesis_error": self.synthesis_error, # Added to avoid KeyError in inference.py
            "metadata": {
                "critical_path_depth": getattr(self, 'baseline_critical_path', 0),
                "total_ops": getattr(self, 'baseline_total_ops', 0),
                "max_area": self.MAX_AREA
            }
        }

    def _is_valid_schedule_action(self, node_id: int) -> bool:
        if node_id not in self.hls_state.nodes: return False
        node = self.hls_state.nodes[node_id]
        if node.state != State.PENDING: return False
        hw = self.hls_state.hardware
        if node.datapath_component == "ALU" and hw.active_alu >= hw.total_alu: return False
        if node.datapath_component == "MAC" and hw.active_mac >= hw.total_mac: return False
        if node.datapath_component == "MEM" and hw.active_mem >= hw.total_mem: return False
        for parent in self.dag.predecessors(node_id):
            if self.hls_state.nodes[parent].state != State.COMPLETED: return False
        return True

    def _update_physics(self):
        self.hls_state.current_cycle += 1
        nodes_completed_this_cycle = 0
        hw = self.hls_state.hardware
        for node in self.hls_state.nodes.values():
            if node.state == State.EXECUTING:
                node.cycles_remaining -= 1
                if node.cycles_remaining <= 0:
                    node.state = State.COMPLETED
                    node.cycles_remaining = 0
                    nodes_completed_this_cycle += 1
                    if node.datapath_component == "ALU": hw.active_alu -= 1
                    elif node.datapath_component == "MAC": hw.active_mac -= 1
                    elif node.datapath_component == "MEM": hw.active_mem -= 1
        return nodes_completed_this_cycle

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        nodes_completed = 0
        
        # --- PARETO BUDGET CHECK ---
        if self.current_area > self.MAX_AREA:
            # Over budget catastrophic termination
            reward = -100.0
            terminated = True
            return self._get_obs(), reward, terminated, truncated, self._get_info()
            
        # Parse Flattened Action
        is_button = (action == self.BUTTON_ACTION)
        
        if self.phase == "TRANSFORM":
            if is_button:
                # DONE_TRANSFORMING exactly commits the graph
                self.phase = "SCHEDULE"
                self._recompute_criticality()
                self._init_hls_state()
            else:
                node_id = action // self.PRAGMAS
                pragma = action % self.PRAGMAS
                if pragma == 0:
                    reward = -10.0 # Penalized for attempting to schedule mid-transform
                elif node_id >= self.real_nodes:
                    reward = -10.0
                else:
                    success = self._apply_pragma(node_id, pragma)
                    if not success:
                        reward = -1.0 # Invalid Pragma Application
                    else:
                        reward = 0.5 # Small bonus for creatively adding pragmas
        
        elif self.phase == "SCHEDULE":
            if is_button:
                nodes_completed = self._update_physics()
                reward -= 1.0 
            else:
                node_id = action // self.PRAGMAS
                pragma = action % self.PRAGMAS
                
                if pragma != 0:
                    reward = -10.0 # Penalized for transforming while locked
                elif not self._is_valid_schedule_action(node_id):
                    reward = -10.0
                    terminated = True
                else:
                    node = self.hls_state.nodes[node_id]
                    node.state = State.EXECUTING
                    node.cycles_remaining = node.latency
                    
                    self.execution_history.append({
                        "node_id": node_id,
                        "op": node.op_type,
                        "resource": node.datapath_component,
                        "start_cycle": self.hls_state.current_cycle,
                        "end_cycle": self.hls_state.current_cycle + node.latency
                    })
                    
                    hw = self.hls_state.hardware
                    if node.datapath_component == "ALU": hw.active_alu += 1
                    elif node.datapath_component == "MAC": hw.active_mac += 1
                    elif node.datapath_component == "MEM": hw.active_mem += 1
            
            reward += (0.2 * nodes_completed)
            all_completed = all(n.state == State.COMPLETED for n in self.hls_state.nodes.values())
            if all_completed:
                # The Golden Carrot: Overrides all accumulated wait penalties
                reward += 100.0
                terminated = True
                
        return self._get_obs(), reward, terminated, truncated, self._get_info()
