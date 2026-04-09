import os
import textwrap
from typing import List, Optional
from openai import OpenAI
from env import LlamaRouterEnv
from tasks import get_task_and_grader
from dotenv import load_dotenv

load_dotenv()  

# --- SCALER MANDATORY HELPER FUNCTIONS ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_inference():
    # Use the 2026 router and free model suffix
    API_BASE_URL = "https://router.huggingface.co/v1"
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct:hf-inference"
    
    # Using your working environment variable or split token method
    HF_TOKEN = os.getenv("HF_TOKEN") 

    if not HF_TOKEN:
        print("❌ ERROR: HF_TOKEN not found. Check your Space Secrets!", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
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
                user_prompt = f"Observation: [Length={obs[0]:.2f}, Complexity={obs[1]:.2f}, Budget={obs[2]:.2f}]. Action:"
                
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                        max_tokens=5,
                        temperature=0.0
                    )
                    action_str = response.choices[0].message.content.strip()
                    # Extract the first digit found
                    action = int(''.join(filter(str.isdigit, action_str))[0])
                except Exception as e:
                    action = 2 # Fallback
                    print(f"AI Error: {e}", flush=True)
                    
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                rewards.append(float(reward))
                log_step(step=step_count, action=str(action), reward=reward, done=done, error=None)
                
            final_score = grader.grade(info)
            log_end(success=(final_score > 0.0), steps=step_count, score=final_score, rewards=rewards)
            env.close()
        except Exception as e:
            print(f"Loop Error: {e}", flush=True)

# 🚀 CRITICAL FIX: No server code here! Just run the script directly.
if __name__ == "__main__":
    run_inference()