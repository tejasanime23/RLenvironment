from pydantic import BaseModel, Field
from typing import List

# --- Observation Schema ---
class HLSObservation(BaseModel):
    """
    The state of the HLS scheduling environment at a given clock cycle.
    """
    node_features: List[List[float]] = Field(
        ..., description="Padded 300x6 matrix containing OpCode, Status, Latency, In-Degree, Criticality, and Pragma State."
    )
    adj_matrix: List[List[float]] = Field(
        ..., description="Padded 300x300 adjacency matrix representing Data and Control dependencies."
    )
    global_state: List[float] = Field(
        ..., description="Vector: [Current Cycle, Busy Resources, Progress Ratio, Area Ratio, Ready Node Count]."
    )
    action_mask: List[int] = Field(
        ..., description="Binary mask indicating which actions are legally available (Node*6 + Pragma)."
    )

# --- Action Schema ---
class HLSAction(BaseModel):
    """
    A flattened discrete action representing both the Node selection and the Pragma transformation.
    action_id = (Node_ID * 6) + Pragma_Type. 
    The maximum value acts as the WAIT/DONE_TRANSFORMING command.
    """
    action_id: int = Field(
        ..., description="Integer representing the specific node and transformation/schedule command.", ge=0, le=(300 * 6) + 1
    )
