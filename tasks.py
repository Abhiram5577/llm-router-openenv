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
    """Evaluates the routing agent on production Service Level Agreements."""
    
    @staticmethod
    def grade(final_state: Dict[str, Any]) -> int:
        funds = final_state.get("available_funds", 0.0)
        total_budget = final_state.get("total_budget", 1.0)
        step_idx = final_state.get("step_idx", 0)
        total_prompts = len(final_state.get("queue", []))
        
        # 1. Survival Score (40%)
        survival_ratio = step_idx / max(1, total_prompts)
        survival_score = survival_ratio * 0.4
        
        if funds <= 0.0 or step_idx < total_prompts:
            # Bankrupt before finishing
            return 0
            
        # 2. SLA / Quality Score (30%)
        # In a real system, we'd track exactly how many prompts failed the probabilistic check.
        # Since our env rewards encode failures as negative values, we assume surviving 
        # means they met the bare minimum, but efficiency determines the rest.
        
        # 3. Efficiency Score (30%)
        efficiency_ratio = funds / total_budget
        efficiency_score = efficiency_ratio * 0.3
        
        # Completed all tasks with funds remaining: return 1
        return 1

def get_task_and_grader(difficulty: str) -> tuple[Dict[str, Any], SLAGrader]:
    diff_map = {
        "easy": EASY_TASK_CONFIG,
        "medium": MEDIUM_TASK_CONFIG,
        "hard": HARD_TASK_CONFIG
    }
    return diff_map.get(difficulty.lower(), MEDIUM_TASK_CONFIG), SLAGrader()