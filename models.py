from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any

class State(Enum):
    PENDING = 0
    EXECUTING = 1
    COMPLETED = 2

@dataclass
class NodeState:
    node_id: int
    op_type: str
    latency: int
    state: State = State.PENDING
    cycles_remaining: int = 0
    
    # Hardware Annotation 
    datapath_component: str = "NONE" # (e.g. ALU, MAC, MEM)
    target_register: str = ""        # (e.g. R1, R2)

@dataclass
class HardwareLimits:
    # How many of each physical component exist on the board (Phase 3 Constraints)
    total_alu: int = 2
    total_mac: int = 1
    total_mem: int = 1
    
    # Active tally during execution
    active_alu: int = 0
    active_mac: int = 0
    active_mem: int = 0

@dataclass
class HLSState:
    current_cycle: int = 0
    nodes: Dict[int, NodeState] = field(default_factory=dict)
    hardware: HardwareLimits = field(default_factory=HardwareLimits)
