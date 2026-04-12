import os
import textwrap
from typing import List, Optional
from openai import OpenAI
from env import LlamaRouterEnv
from tasks import get_task_and_grader
from dotenv import load_dotenv

# Environment variables are injected by the platform; never override them
load_dotenv(override=False)  

# --- SCALER MANDATORY HELPER FUNCTIONS ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str], obs: list = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    
    action_map = {"0": "1B_Fast", "1": "8B_Core", "2": "70B_Pro"}
    action_readable = action_map.get(action, action)
    
    latency = obs[3] * 5.0 if obs else 0.0
    latency_status = "WARNING: SPIKE" if latency > 2.5 else "OK"
    
    print(f"[STEP] step={step} route={action_readable} latency={latency:.2f}s {latency_status} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_inference():
    API_BASE_URL = os.environ.get("API_BASE_URL")
    API_KEY = os.environ.get("API_KEY")
    MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct:hf-inference")

    if not API_BASE_URL or not API_KEY:
        print("ERROR: API_BASE_URL or API_KEY not found in environment!", flush=True)
        return

    try:
        client = OpenAI(
            base_url=API_BASE_URL, 
            api_key=API_KEY,
            max_retries=3
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI client: {e}", flush=True)
        return

    tasks = ["easy", "medium", "hard"]
    
    for task in tasks:
        try:
            config, grader = get_task_and_grader(task)
            env = LlamaRouterEnv(config=config)
            log_start(task=task, env="llama_router", model=MODEL_NAME)
            
            obs, info = env.reset()
            done = False
            step_count = 0
            rewards = []
            
            while not done:
                step_count += 1
                system_prompt = "You are an AI Inference Gateway. Reply with ONLY 0, 1, or 2."
                user_prompt = f"Observation: [Length={obs[0]:.2f}, Complexity={obs[1]:.2f}, Budget={obs[2]:.2f}, Latency={obs[3]:.2f}, Failures={obs[4]:.2f}]. Action:"
                
                action = 1
                error_msg = None
                
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                        max_tokens=5,
                        temperature=0.0,
                        timeout=30  # <-- ADD THIS
                    )
                    action_str = response.choices[0].message.content.strip()
                    try:
                        action = int(''.join(filter(str.isdigit, action_str))[0])
                    except (IndexError, ValueError) as parse_err:
                        action = 1
                        error_msg = f"Parse error: {parse_err}"
                        print(f"Parse error: {error_msg}", flush=True)
                except Exception as api_err:
                    action = 1
                    error_msg = f"API error: {api_err}"
                    print(f"API error: {error_msg}", flush=True)
                    
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                rewards.append(float(reward))
                log_step(step=step_count, action=str(action), reward=reward, done=done, error=error_msg, obs=list(obs))
                
            final_score = grader.grade(info)
            log_end(success=(final_score > 0.0), steps=step_count, score=final_score, rewards=rewards)
            env.close()
        except Exception as e:
            print(f"Task error ({task}): {e}", flush=True)

if __name__ == "__main__":
    run_inference()