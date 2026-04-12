from pydantic import BaseModel, Field
from enum import IntEnum
from typing import List, Optional

class LlamaVariant(IntEnum):
    LLAMA_1B = 0   # Llama 3.2 1B (Ultra-fast, specialized for simple extraction)
    LLAMA_8B = 1   # Llama 3.1 8B (The workhorse)
    LLAMA_70B = 2  # Llama 3.1 70B (High-reasoning, expensive)

class PromptMetadata(BaseModel):
    token_count: int = Field(..., ge=1)
    complexity_score: float = Field(..., ge=0.0, le=1.0)
    priority: int = Field(default=1, description="1=Normal, 2=Urgent (SLA is tighter)")

class GatewayState(BaseModel):
    queue: List[PromptMetadata]
    step_idx: int = 0
    available_funds: float
    total_budget: float
    accumulated_latency: float = 0.0
    previous_failures: int = 0
    simulated_70b_latency: float = 0.0
    
    @property
    def is_exhausted(self) -> bool:
        return self.step_idx >= len(self.queue) or self.available_funds <= 0