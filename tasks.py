from typing import Dict, Any

# ---------------------------------------------------------
# Task Configurations
# ---------------------------------------------------------
# We define three difficulty tiers by adjusting the traffic 
# volume and the available budget.

EASY_TASK_CONFIG = {
    "num_prompts": 20,         # Short queue
    "initial_budget": 5.0,     # Massive budget ($5.00). Agent can just spam LARGE model.
    "task_id": "llm-router-easy"
}

MEDIUM_TASK_CONFIG = {
    "num_prompts": 50,         # Standard queue
    "initial_budget": 1.0,     # Standard budget ($1.00). Agent must balance choices.
    "task_id": "llm-router-medium"
}

HARD_TASK_CONFIG = {
    "num_prompts": 50,         # Standard queue
    "initial_budget": 0.3,     # Starvation budget ($0.30). Agent must perfectly optimize.
    "task_id": "llm-router-hard"
}

# ---------------------------------------------------------
# OpenEnv Grader
# ---------------------------------------------------------
class RouterGrader:
    """
    Evaluates the final state of the LLMRouterEnv and returns a normalized
    score between 0.0 and 1.0.
    """
    
    @staticmethod
    def grade(final_state: Dict[str, Any]) -> float:
        """
        Calculates the score based on survival and budget efficiency.
        
        Args:
            final_state (dict): The dictionary returned by env.state() at the end of the episode.
            
        Returns:
            float: Score strictly between 0.0 and 1.0.
        """
        # 1. Extract data from the state dictionary
        remaining_budget = final_state.get("remaining_budget", 0.0)
        total_budget = final_state.get("total_budget", 1.0)
        current_step = final_state.get("current_step", 0)
        total_prompts = len(final_state.get("prompts", []))
        
        # 2. Check for failure (Bankruptcy)
        if remaining_budget <= 0.0 or current_step < total_prompts:
            # If they went bankrupt before finishing the queue, score is heavily penalized.
            # We give a partial score based on how many prompts they survived.
            survival_rate = current_step / max(1, total_prompts)
            raw_score = survival_rate * 0.4 # Cap failure score at 0.4
            
        else:
            # 3. Calculate Success Score
            # They survived! The score is 0.5 (for surviving) plus up to 0.5 for efficiency.
            # Efficiency is how much budget they have left over.
            efficiency_ratio = remaining_budget / total_budget
            raw_score = 0.5 + (0.5 * efficiency_ratio)
            
        # 4. Strict OpenEnv bounds enforcement [0.0, 1.0]
        final_score = max(0.0, min(1.0, raw_score))
        
        return final_score

# Helper function to easily grab a task
def get_task_and_grader(difficulty: str) -> tuple[Dict[str, Any], RouterGrader]:
    """Returns the config and grader for a specific difficulty tier."""
    if difficulty.lower() == "easy":
        return EASY_TASK_CONFIG, RouterGrader()
    elif difficulty.lower() == "hard":
        return HARD_TASK_CONFIG, RouterGrader()
    else:
        return MEDIUM_TASK_CONFIG, RouterGrader()