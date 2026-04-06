import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from schemas import LlamaVariant, GatewayState, PromptMetadata

class LlamaRouterEnv(gym.Env):
    """
    Simulates a production inference gateway for the Llama 3 ecosystem.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        
        # Action: Route to 1B, 8B, or 70B
        self.action_space = spaces.Discrete(3)
        # Obs: [norm_len, complexity, norm_funds, priority]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Real-world Llama pricing (approx $ per 1M tokens)
        self.costs = {LlamaVariant.LLAMA_1B: 0.0001, LlamaVariant.LLAMA_8B: 0.0003, LlamaVariant.LLAMA_70B: 0.0015}
        self.capabilities = {LlamaVariant.LLAMA_1B: 0.25, LlamaVariant.LLAMA_8B: 0.75, LlamaVariant.LLAMA_70B: 1.0}

    def _get_obs(self):
        p = self.state.queue[self.state.step_idx]
        return np.array([
            p.token_count / 2048,
            p.complexity_score,
            self.state.available_funds / self.state.total_budget,
            p.priority / 2.0
        ], dtype=np.float32)

    def step(self, action):
        variant = LlamaVariant(action)
        prompt = self.state.queue[self.state.step_idx]
        
        # 1. Cost with 'Network Jitter'
        actual_cost = prompt.token_count * self.costs[variant] * random.uniform(0.95, 1.05)
        self.state.available_funds -= actual_cost
        
        if self.state.available_funds <= 0:
            return np.zeros(4), -2.0, True, False, self.state.model_dump()

        # 2. Quality check (Probabilistic)
        # Even a 70B can fail a 1.0 complexity task 2% of the time.
        success_threshold = self.capabilities[variant]
        is_successful = random.random() < (success_threshold / max(0.01, prompt.complexity_score))
        
        # 3. Reward: Efficiency + Quality
        if not is_successful:
            reward = -1.0 if prompt.priority == 2 else -0.5
        else:
            # Reward for saving money: (Max Cost - Actual Cost) / Max Cost
            max_cost = prompt.token_count * self.costs[LlamaVariant.LLAMA_70B]
            efficiency = (max_cost - actual_cost) / max_cost
            reward = 0.2 + (0.8 * efficiency)

        self.state.step_idx += 1
        done = self.state.is_exhausted
        
        return self._get_obs() if not done else np.zeros(4), float(reward), done, False, self.state.model_dump()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed: random.seed(seed)
        
        # Generate dynamic scenarios
        num_prompts = self.config.get("prompts", 50)
        scenario = self.config.get("scenario", "normal")
        
        queue = []
        for i in range(num_prompts):
            # Modeling 'Traffic Waves'
            if scenario == "burst" and 10 < i < 20:
                comp = random.uniform(0.8, 1.0) # High complexity burst
                pri = 2
            else:
                comp = random.uniform(0.1, 0.7)
                pri = 1
            queue.append(PromptMetadata(token_count=random.randint(128, 2048), complexity_score=comp, priority=pri))
            
        self.state = GatewayState(queue=queue, available_funds=self.config.get("budget", 1.0), total_budget=self.config.get("budget", 1.0))
        return self._get_obs(), self.state.model_dump()