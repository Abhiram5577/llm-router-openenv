import gradio as gr
from typing import Optional, Tuple
from env import LLMRouterEnv
from tasks import get_task_and_grader

# Global state management (since Gradio can't serialize Gym environments)
# We store difficulty + environment state separately
current_env: Optional[LLMRouterEnv] = None
current_difficulty: str = "medium"


def start_episode(difficulty: str) -> Tuple[str, str, str, str, str]:
    """Initializes the environment based on selected difficulty."""
    global current_env, current_difficulty
    
    current_difficulty = difficulty
    config, _ = get_task_and_grader(difficulty)
    current_env = LLMRouterEnv(config=config)
    obs, info = current_env.reset()
    
    return update_ui_display()


def step_episode(action_idx: int) -> Tuple[str, str, str, str, str]:
    """Takes a single action in the environment and updates the UI."""
    global current_env
    
    if current_env is None or current_env.env_state.is_done:
        return (
            "🛑 Episode finished! Please reset.",
            "$0.00 / $0.00",
            "Queue Empty",
            "Progress: 0/0",
            "Please start a new episode"
        )
    
    # Execute the action
    obs, reward, terminated, truncated, info = current_env.step(action_idx)
    
    return update_ui_display()


def update_ui_display() -> Tuple[str, str, str, str, str]:
    """Extract environment state and format UI elements."""
    global current_env
    
    if current_env is None:
        return (
            "Waiting to start...",
            "$0.00 / $0.00",
            "No prompt",
            "Progress: 0/0",
            "Please select difficulty and click Start"
        )
    
    state = current_env.state()
    is_done = current_env.env_state.is_done or state.get("remaining_budget", 0) <= 0
    
    # Status message
    if is_done:
        status = "🛑 EPISODE OVER - Click Start to begin a new episode"
    else:
        status = "✅ Current prompt ready - Select a model"
    
    # Budget display
    budget_text = f"${state.get('remaining_budget', 0):.4f} / ${state.get('total_budget', 1):.2f}"
    
    # Current prompt info
    if not is_done and state.get("prompts"):
        current_prompt = state["prompts"][state["current_step"]]
        prompt_text = f"**Tokens:** {current_prompt['length_tokens']} | **Complexity:** {current_prompt['complexity']:.2f}/1.0"
    else:
        prompt_text = "Queue Empty or Bankrupt - Episode Complete"
    
    # Progress
    progress_text = f"Prompt {state.get('current_step', 0)} of {len(state.get('prompts', []))}"
    
    # Last reward (if available)
    if state.get('current_step', 0) > 0 and not is_done:
        rewards_info = "Ready for next action"
    else:
        rewards_info = "Episode in progress"
    
    return status, budget_text, prompt_text, progress_text, rewards_info


# ═══════════════════════════════════════════════════════════════
# GRADIO UI LAYOUT
# ═══════════════════════════════════════════════════════════════
with gr.Blocks(title="LLM Router OpenEnv", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🚦 LLM Gateway Router (OpenEnv Demo)

You are the AI routing agent. For each incoming prompt, select which model to use:
- **Small Model (Cheap)**: Low cost, works for simple tasks
- **Medium Model (Balanced)**: Moderate cost, handles most tasks
- **Large Model (Expensive)**: High cost, handles everything

**Goal**: Process all prompts while staying within budget and maintaining quality!
    """)
    
    # Control Section
    with gr.Row():
        with gr.Column():
            difficulty_dropdown = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="medium",
                label="🎯 Task Difficulty",
                info="Easy: $5 budget. Medium: $1 budget. Hard: $0.30 budget."
            )
            start_btn = gr.Button("▶️ Start / Reset Episode", variant="primary", size="lg")
            
        with gr.Column():
            status_display = gr.Markdown("### Status\nWaiting to start...")
            budget_display = gr.Markdown("### Budget\n$0.00 / $0.00")
    
    # Current Prompt Section
    gr.Markdown("### 📩 Current Incoming Prompt")
    prompt_display = gr.Markdown("No prompt yet")
    progress_display = gr.Markdown("Progress: 0/0")
    
    # Action Buttons
    gr.Markdown("### 🧠 Select Routing Action")
    with gr.Row():
        btn_small = gr.Button("💰 Small Model (Cheap)", scale=1)
        btn_medium = gr.Button("⚖️ Medium Model (Balanced)", scale=1)
        btn_large = gr.Button("🚀 Large Model (Expensive)", scale=1)
    
    reward_display = gr.Markdown("### Info\nReady for next decision")
    
    gr.Markdown("""
---
### How Scoring Works:
- **Survive**: Process all prompts without running out of budget
- **Quality**: Select appropriate models (avoid cheap models for complex tasks)
- **Efficiency**: Save budget by using cheap models when suitable
- **Score**: 0.0 (failed) to 1.0 (perfect)
    """)
    
    # Event Bindings
    start_btn.click(
        fn=start_episode,
        inputs=[difficulty_dropdown],
        outputs=[status_display, budget_display, prompt_display, progress_display, reward_display],
        queue=False
    )
    
    # Action buttons - pass action index (0, 1, 2)
    btn_small.click(
        fn=lambda: step_episode(0),
        outputs=[status_display, budget_display, prompt_display, progress_display, reward_display],
        queue=False
    )
    
    btn_medium.click(
        fn=lambda: step_episode(1),
        outputs=[status_display, budget_display, prompt_display, progress_display, reward_display],
        queue=False
    )
    
    btn_large.click(
        fn=lambda: step_episode(2),
        outputs=[status_display, budget_display, prompt_display, progress_display, reward_display],
        queue=False
    )


if __name__ == "__main__":
    # Server configuration explicitly for Hugging Face Spaces / Docker
    print("\n" + "="*60)
    print("🌐 LLM Router OpenEnv - Web UI")
    print("="*60)
    print("\n🚀 Server starting...")
    print("📍 Open browser to: http://localhost:7860")
    print("\n" + "="*60 + "\n")
    
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False,
        show_error=True
    )