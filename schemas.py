from pydantic import BaseModel, Field
from enum import IntEnum
from typing import List

# ---------------------------------------------------------
# Action Space Definition
# ---------------------------------------------------------
class ModelChoice(IntEnum):
    """The 3 discrete actions available to the agent."""
    SMALL = 0   # Cheap, low capability (e.g., 8B parameter model)
    MEDIUM = 1  # Moderate cost, moderate capability
    LARGE = 2   # Expensive, high capability (e.g., 70B parameter model)

# ---------------------------------------------------------
# Internal State Definitions (Hidden from Agent)
# ---------------------------------------------------------
class Prompt(BaseModel):
    """Represents a single user request in the queue."""
    length_tokens: int = Field(..., ge=10, le=2000, description="Raw token count of the prompt")
    complexity: float = Field(..., ge=0.0, le=1.0, description="0.0 is trivial, 1.0 is highly complex")

class EnvState(BaseModel):
    """The complete internal state of the environment."""
    prompts: List[Prompt] = Field(default_factory=list, description="Queue of incoming prompts")
    current_step: int = Field(0, ge=0, description="Index of the current prompt being processed")
    remaining_budget: float = Field(..., description="Remaining budget in dollars")
    total_budget: float = Field(..., description="Initial budget in dollars, used for normalization")
    
    @property
    def is_done(self) -> bool:
        """Helper to check if we've processed the entire queue."""
        return self.current_step >= len(self.prompts)

# ---------------------------------------------------------
# Observation Space Definition (Visible to Agent)
# ---------------------------------------------------------
class Observation(BaseModel):
    """The normalized telemetry data given to the agent at each step."""
    norm_length: float = Field(..., ge=0.0, le=1.0, description="Prompt length scaled 0 to 1")
    complexity: float = Field(..., ge=0.0, le=1.0, description="Prompt complexity (already 0 to 1)")
    norm_budget: float = Field(..., ge=0.0, le=1.0, description="Remaining budget scaled 0 to 1")
    
    def to_numpy(self):
        """Helper to convert Pydantic model to a flat Numpy array for the RL agent."""
        import numpy as np
        return np.array([self.norm_length, self.complexity, self.norm_budget], dtype=np.float32)