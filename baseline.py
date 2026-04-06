import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Import our custom OpenEnv files
from env import LlamaRouterEnv
from tasks import get_task_and_grader

def run_baseline():
    """Trains a baseline PPO agent and evaluates it on the Medium task."""
    
    # 1. Load the Task Config and Grader
    difficulty = "medium"
    config, grader = get_task_and_grader(difficulty)
    print(f"--- Loading LLM Router Environment ({difficulty.upper()}) ---")

    # 2. Initialize Environment
    env = LlamaRouterEnv(config=config)

    # 3. Strict API Compliance Check (Crucial for OpenEnv/Hackathons)
    print("Running Stable Baselines 3 Environment Checker...")
    check_env(env, warn=True)
    print("Environment check passed! API is fully compliant.\n")

    # 4. Train a Baseline Agent (PPO is extremely fast and robust for discrete spaces)
    # We use a fixed seed (42) so judges get the exact same reproducible result.
    print("Training PPO Agent (10,000 timesteps)...")
    model = PPO("MlpPolicy", env, verbose=0, seed=42)
    
    # 10k steps takes about 5 seconds on a standard laptop CPU
    model.learn(total_timesteps=10000) 
    print("Training complete!\n")

    # 5. Evaluate the Agent (Inference)
    print("Running Evaluation Episode...")
    obs, info = env.reset(seed=42)
    done = False
    
    # Track metrics for the demo printout
    total_reward = 0.0

    while not done:
        # The trained model predicts the optimal routing choice
        action, _states = model.predict(obs, deterministic=True)

        # Step the environment forward
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        done = terminated or truncated

    # 6. Grade the final state using the OpenEnv standard Grader
    final_score = grader.grade(info)
    
    print("====================================")
    print("         EVALUATION RESULTS         ")
    print("====================================")
    print(f"Task ID:          {config['task_id']}")
    print(f"Cumulative Reward: {total_reward:.2f}")
    print(f"Budget Remaining: ${info.get('available_funds', 0):.2f} / ${info.get('total_budget', config['budget']):.2f}")
    print(f"Prompts Routed:   {info.get('step_idx', 0)} / {len(info.get('queue', []))}")
    print(f"OpenEnv Score:    {final_score:.2f} / 1.00")
    print("====================================")

if __name__ == "__main__":
    run_baseline()