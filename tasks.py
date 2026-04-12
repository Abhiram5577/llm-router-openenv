from typing import Dict, Any

EASY_TASK_CONFIG = {
    "prompts": 20,
    "budget": 5.0,
    "scenario": "normal", # Steady, predictable traffic
    "task_id": "llama-router-easy"
}

MEDIUM_TASK_CONFIG = {
    "prompts": 50,
    "budget": 1.5,
    "scenario": "normal", # Mixed traffic
    "task_id": "llama-router-medium"
}

HARD_TASK_CONFIG = {
    "prompts": 50,
    "budget": 0.4,
    "scenario": "burst", # Massive mid-episode complexity surge
    "task_id": "llama-router-hard"
}

class SLAGrader:
    """Maps final state to a continuous score in (0.01, 0.99]."""
    
    @staticmethod
    def grade(final_state: Dict[str, Any]) -> float:
        funds = final_state.get("available_funds", 0.0)
        total_budget = final_state.get("total_budget", 1.0)
        step_idx = final_state.get("step_idx", 0)
        queue = final_state.get("queue", [])
        total_prompts = len(queue)
        
        if funds <= 0.0 or step_idx < total_prompts:
            return 0.01
        
        total_tokens_used = sum(p.get("token_count", 0) for p in queue[:step_idx])
        survival_ratio = step_idx / max(1, total_prompts)
        base_quality_score = 0.1 + (0.8 * survival_ratio)
        tokens_penalty = (total_tokens_used / 2048.0) * 0.15
        efficiency_ratio = funds / total_budget
        efficiency_bonus = efficiency_ratio * 0.1
        continuous_score = base_quality_score - tokens_penalty + efficiency_bonus
        final_score = max(0.01, min(0.99, continuous_score))
        
        return final_score

def get_task_and_grader(difficulty: str) -> tuple[Dict[str, Any], SLAGrader]:
    diff_map = {
        "easy": EASY_TASK_CONFIG,
        "medium": MEDIUM_TASK_CONFIG,
        "hard": HARD_TASK_CONFIG
    }
    return diff_map.get(difficulty.lower(), MEDIUM_TASK_CONFIG), SLAGrader()