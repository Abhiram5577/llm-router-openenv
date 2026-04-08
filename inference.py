from dotenv import load_dotenv
load_dotenv()  
import os
import textwrap
from openai import OpenAI
from env import LlamaRouterEnv
from tasks import get_task_and_grader

def run_inference():
    print("🚀 Booting Meta OpenEnv Inference via Hugging Face Router...")
    
    # Reads the Hugging Face token from the OPENAI_API_KEY environment variable
    # This is the exact hackathon requirement from the dashboard!
    api_key = os.environ.get("OPENAI_API_KEY") 
    
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set.")
        print("Run this in your terminal first: export OPENAI_API_KEY='your_hf_token'")
        return

    # Initialize client pointed at Hugging Face (Dashboard Spec)
    client = OpenAI(
        base_url="https://api-inference.huggingface.co/v1/",
        api_key=api_key
    )
    
    # Using Llama 3 8B Instruct as our "Agent Brain"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct" 
    
    # Loop through our 3 defined tasks
    tasks = ["easy", "medium", "hard"]
    
    for task in tasks:
        print(f"\n--- Running Task: {task.upper()} ---")
        config, grader = get_task_and_grader(task)
        env = LlamaRouterEnv(config=config)
        
        obs, info = env.reset()
        done = False
        
        while not done:
            # The System Prompt guiding the LLM
            system_prompt = textwrap.dedent("""
                You are an AI Inference Gateway. Route the prompt to the most efficient model based on budget and complexity.
                0: Llama 3.2 1B (Cheap, use for complexity < 0.25)
                1: Llama 3.1 8B (Balanced, use for complexity < 0.75)
                2: Llama 3.1 70B (Expensive, use for complexity > 0.75 or if budget is high)
                Reply ONLY with a single integer: 0, 1, or 2.
            """).strip()
            
            user_prompt = f"Observation: [Length={obs[0]:.2f}, Complexity={obs[1]:.2f}, Budget={obs[2]:.2f}]. Action:"
            
            try:
                # Call the Hugging Face Router
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=5,
                    temperature=0.0
                )
                action_str = response.choices[0].message.content.strip()
                
                # Safely extract the integer action
                action = int(''.join(filter(str.isdigit, action_str))[0])
                if action not in [0, 1, 2]: action = 2
            except Exception as e:
                # Fallback to Large model if API fails
                action = 2
                
            # Step the environment forward
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        # Grade the final state
        final_score = grader.grade(info)
        print(f"✅ Task {task.upper()} completed. Final OpenEnv Score: {final_score:.2f} / 1.00")

if __name__ == "__main__":
    run_inference()