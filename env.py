import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from schemas import LlamaVariant, GatewayState, PromptMetadata

class LlamaRouterEnv(gym.Env):
    """
    Production inference gateway. Handles multi-step escalation and continuous rewards.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        
        self.action_space = spaces.Discrete(3)  # 0=1B, 1=8B, 2=70B
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        
        # Reflect production tier costs
        self.costs = {LlamaVariant.LLAMA_1B: 0.0001, LlamaVariant.LLAMA_8B: 0.0003, LlamaVariant.LLAMA_70B: 0.0015}
        self.capabilities = {LlamaVariant.LLAMA_1B: 0.25, LlamaVariant.LLAMA_8B: 0.75, LlamaVariant.LLAMA_70B: 1.0}

    def _get_obs(self):
        p = self.state.queue[self.state.step_idx]
        return np.array([
            p.token_count / 2048,
            p.complexity_score,
            self.state.available_funds / self.state.total_budget,
            self.state.simulated_70b_latency / 5.0,  # [0,1]
            self.state.previous_failures / 10.0  # [0,1]
        ], dtype=np.float32)

    def step(self, action):
        if action not in [0, 1, 2]:
            # Punish hallucinations: invalid action terminates with -1.0
            return np.zeros(5), -1.0, True, False, self.state.model_dump()
        
        variant = LlamaVariant(action)
        prompt = self.state.queue[self.state.step_idx]
        
        # Simulate cost variance (spot-instance pricing, variable token generation)
        actual_cost = prompt.token_count * self.costs[variant] * random.uniform(0.95, 1.05)
        self.state.available_funds -= actual_cost
        
        if self.state.available_funds <= 0:
            return np.zeros(5), -2.0, True, False, self.state.model_dump()

        # Models fail stochastically; even 70B can miss hard tasks
        success_threshold = self.capabilities[variant]
        is_successful = random.random() < (success_threshold / max(0.01, prompt.complexity_score))
        
        if not is_successful and action in [0, 1]:
            # Cheap model failed: force retry without advancing. Penalize escalation cost.
            self.state.previous_failures += 1
            escalation_penalty = 0.1 * actual_cost
            self.state.available_funds -= escalation_penalty
            done = False
            reward = -0.3 - (escalation_penalty / max(0.01, self.state.total_budget))
            
            return self._get_obs(), float(reward), done, False, self.state.model_dump()
        
        if not is_successful:
            reward = -1.0 if prompt.priority == 2 else -0.5
        else:
            quality_score = 0.1 + (0.8 * success_threshold)
            tier_cost_multiplier = self.costs[variant] / self.costs[LlamaVariant.LLAMA_70B]
            tokens_used_penalty = (prompt.token_count / 2048.0) * tier_cost_multiplier
            reward = quality_score - tokens_used_penalty
        
        self.state.previous_failures = 0
        self.state.step_idx += 1
        done = self.state.is_exhausted
        
        return self._get_obs() if not done else np.zeros(5), float(reward), done, False, self.state.model_dump()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed: random.seed(seed)
        
        num_prompts = self.config.get("prompts", 50)
        scenario = self.config.get("scenario", "normal")
        
        queue = []
        for i in range(num_prompts):
            if scenario == "burst" and 10 < i < 20:
                comp = random.uniform(0.8, 1.0)
                pri = 2
            else:
                comp = random.uniform(0.1, 0.7)
                pri = 1
            queue.append(PromptMetadata(token_count=random.randint(128, 2048), complexity_score=comp, priority=pri))
        
        simulated_latency = random.uniform(0.5, 4.0)
        
        self.state = GatewayState(
            queue=queue, 
            available_funds=self.config.get("budget", 1.0), 
            total_budget=self.config.get("budget", 1.0),
            previous_failures=0,
            simulated_70b_latency=simulated_latency
        )
        return self._get_obs(), self.state.model_dump()