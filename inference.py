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
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def run_inference():
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN") 

    if not HF_TOKEN:
        print("❌ ERROR: HF_TOKEN environment variable not set. Add it to your Space Secrets!")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    tasks = ["easy", "medium", "hard"]
    
    for task in tasks:
        config, grader = get_task_and_grader(task)
        env = LlamaRouterEnv(config=config)
        
        # 1. SCALER START LOG
        log_start(task=task, env="llama_router", model=MODEL_NAME)
        
        obs, info = env.reset()
        done = False
        step_count = 0
        rewards = []
        
        while not done:
            step_count += 1
            system_prompt = textwrap.dedent("""
                You are an AI Inference Gateway. Route the prompt to the most efficient model based on budget and complexity.
                0: Llama 3.2 1B (Cheap, use for complexity < 0.25)
                1: Llama 3.1 8B (Balanced, use for complexity < 0.75)
                2: Llama 3.1 70B (Expensive, use for complexity > 0.75 or if budget is high)
                Reply ONLY with a single integer: 0, 1, or 2.
            """).strip()
            
            user_prompt = f"Observation: [Length={obs[0]:.2f}, Complexity={obs[1]:.2f}, Budget={obs[2]:.2f}]. Action:"
            
            action_str = "2"
            error_msg = None
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=5,
                    temperature=0.0
                )
                action_str = response.choices[0].message.content.strip()
                action = int(''.join(filter(str.isdigit, action_str))[0])
                if action not in [0, 1, 2]: action = 2
            except Exception as e:
                action = 2
                error_msg = str(e)
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            rewards.append(float(reward))

            # 2. SCALER STEP LOG
            log_step(step=step_count, action=action_str, reward=reward, done=done, error=error_msg)
            
        final_score = grader.grade(info)
        # Assume success if score is greater than 0
        success = final_score > 0.0 
        
        # 3. SCALER END LOG
        log_end(success=success, steps=step_count, score=final_score, rewards=rewards)
        
        try:
            env.close()
        except:
            pass

if __name__ == "__main__":
    run_inference()