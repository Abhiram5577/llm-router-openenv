import gymnasium as gym
from stable_baselines3 import PPO
from env import LLMRouterEnv
from tasks import get_task_and_grader

def run_automated_evaluation():
    """
    Automated demo script designed for Hackathon Judges.
    Runs a full RL training and evaluation loop across all three difficulty tiers.
    """
    print("==================================================")
    print("🚀 META OPENENV HACKATHON: AUTOMATED JUDGING SUITE")
    print("==================================================\n")
    
    difficulties = ["easy", "medium", "hard"]
    results = []

    for diff in difficulties:
        print(f"--- [ Booting Task: {diff.upper()} ] ---")
        config, grader = get_task_and_grader(diff)
        env = LLMRouterEnv(config=config)
        
        # Train a fresh, isolated agent for this specific task
        print(f"Training PPO Agent on {diff.upper()} (5,000 steps)...")
        model = PPO("MlpPolicy", env, verbose=0, seed=42)
        model.learn(total_timesteps=5000)
        
        # Evaluate the agent
        print("Evaluating...")
        obs, info = env.reset(seed=42)
        done = False
        total_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
            
        # Grade the final state
        final_score = grader.grade(info)
        
        # Save results for the summary table
        results.append({
            "Tier": diff.upper(),
            "Survived": info.get("current_step", 0) >= len(info.get("prompts", [])),
            "Budget Left": f"${info.get('remaining_budget', 0):.2f}",
            "Score": final_score
        })
        print(f"✅ {diff.upper()} Complete. Score: {final_score:.2f} / 1.00\n")

    # Print Final Summary Table
    print("==================================================")
    print("🏆 FINAL EVALUATION REPORT")
    print("==================================================")
    print(f"{'DIFFICULTY':<12} | {'SURVIVED?':<10} | {'BUDGET LEFT':<12} | {'OPENENV SCORE'}")
    print("-" * 50)
    for res in results:
        survived_str = "Yes" if res["Survived"] else "No"
        print(f"{res['Tier']:<12} | {survived_str:<10} | {res['Budget Left']:<12} | {res['Score']:.2f}")
    print("==================================================")

if __name__ == "__main__":
    run_automated_evaluation()