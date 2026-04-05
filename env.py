import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional, Tuple, Dict, Any, List

# Import the schemas we built in Step 3
from schemas import ModelChoice, Prompt, EnvState, Observation

class LLMRouterEnv(gym.Env):
    """
    OpenEnv compliant reinforcement learning environment for LLM routing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Configuration setup (defaults to medium difficulty)
        self.config = config or {}
        self.num_prompts = self.config.get("num_prompts", 50)
        self.initial_budget = self.config.get("initial_budget", 1.0) # $1.00 budget
        self.max_tokens = 2000
        
        # Action Space: 3 discrete model choices (Small, Medium, Large)
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: [norm_length, complexity, norm_budget] (Strictly 0.0 to 1.0)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Environment Constants defining the "Rules of the World"
        self.MODEL_COSTS = {
            ModelChoice.SMALL: 0.0001,  # $0.0001 per token
            ModelChoice.MEDIUM: 0.0005, # $0.0005 per token
            ModelChoice.LARGE: 0.0015   # $0.0015 per token
        }
        
        self.MODEL_CAPABILITY = {
            ModelChoice.SMALL: 0.3,   # Fails on complexity > 0.3
            ModelChoice.MEDIUM: 0.7,  # Fails on complexity > 0.7
            ModelChoice.LARGE: 1.0    # Handles everything
        }
        
        self.env_state: EnvState = None

    def _generate_traffic(self) -> List[Prompt]:
        """Generates a pseudo-random burst of traffic for the episode."""
        prompts = []
        for _ in range(self.num_prompts):
            length = random.randint(50, self.max_tokens)
            complexity = random.uniform(0.0, 1.0)
            prompts.append(Prompt(length_tokens=length, complexity=complexity))
        return prompts

    def _get_obs(self) -> np.ndarray:
        """Constructs the normalized observation for the agent."""
        current_prompt = self.env_state.prompts[self.env_state.current_step]
        
        obs = Observation(
            norm_length=current_prompt.length_tokens / self.max_tokens,
            complexity=current_prompt.complexity,
            norm_budget=max(0.0, self.env_state.remaining_budget / self.env_state.total_budget)
        )
        return obs.to_numpy()

    def state(self) -> dict:
        """OpenEnv standard: returns the exact hidden state for evaluation/logging."""
        if not self.env_state:
            return {}
        return self.env_state.dict()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        
        # Initialize internal Pydantic state
        self.env_state = EnvState(
            prompts=self._generate_traffic(),
            current_step=0,
            remaining_budget=self.initial_budget,
            total_budget=self.initial_budget
        )
        
        return self._get_obs(), self.state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Executes one routing decision."""
        # Map integer action to our Enum
        choice = ModelChoice(action)
        current_prompt = self.env_state.prompts[self.env_state.current_step]
        
        # 1. Calculate Cost
        cost = current_prompt.length_tokens * self.MODEL_COSTS[choice]
        self.env_state.remaining_budget -= cost
        
        # 2. Check for Budget Bankruptcy (Termination)
        if self.env_state.remaining_budget <= 0:
            # OpenEnv requirement: rewards must be in [0.0, 1.0] range
            reward = 0.0  # Complete failure - no reward
            terminated = True
            return np.zeros(3, dtype=np.float32), reward, terminated, False, self.state()
            
        # 3. Calculate Reward (Quality vs Cost)
        capability = self.MODEL_CAPABILITY[choice]
        
        if capability < current_prompt.complexity:
            # Model is too stupid for this prompt! Hallucination / Bad answer.
            # OpenEnv requirement: rewards in [0.0, 1.0] - failure gets low score
            reward = 0.05
        else:
            # Model successfully answered it.
            # Reward is proportional to how much money we saved by not blindly using LARGE.
            max_possible_cost = current_prompt.length_tokens * self.MODEL_COSTS[ModelChoice.LARGE]
            savings = max_possible_cost - cost
            # Scale reward between 0.1 (used expensive) and 1.0 (used cheap optimally)
            reward = 0.1 + (0.9 * (savings / max_possible_cost))
            
        # 4. Advance Step
        self.env_state.current_step += 1
        
        # 5. Check if Episode is complete (Truncation)
        terminated = False
        truncated = self.env_state.is_done
        
        # If done, return a dummy observation for the final state
        obs = self._get_obs() if not truncated else np.zeros(3, dtype=np.float32)
        
        return obs, float(reward), terminated, truncated, self.state()