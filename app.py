import gradio as gr
import pandas as pd
from env import LlamaRouterEnv
from tasks import get_task_and_grader

# Global state to track history for the charts
history_df = pd.DataFrame(columns=["step", "budget", "model_used"])

def start_episode(difficulty):
    global history_df
    history_df = pd.DataFrame(columns=["step", "budget", "model_used"])
    
    config, _ = get_task_and_grader(difficulty)
    env = LlamaRouterEnv(config=config)
    env.reset()
    return update_ui(env)

def step_episode(env, action_name):
    global history_df
    if env is None or env.state.is_exhausted:
        return [env, "Episode finished!"] + [gr.update()] * 3
    
    action_map = {"Llama 3.2 1B": 0, "Llama 3.1 8B": 1, "Llama 3.1 70B": 2}
    action = action_map[action_name]
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Record telemetry for charts
    new_row = {"step": env.state.step_idx, "budget": env.state.available_funds, "model_used": action_name}
    history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return update_ui(env)

def update_ui(env):
    state = env.state.model_dump()
    done = state["step_idx"] >= len(state["queue"]) or state["available_funds"] <= 0
    
    budget_txt = f"${state['available_funds']:.4f} / ${state['total_budget']:.2f}"
    status = "🛑 EPISODE OVER" if done else f"✅ Active Queue: {state['step_idx']}/{len(state['queue'])}"
    
    # Generate Charts
    line_plot = gr.LinePlot(history_df, x="step", y="budget", title="Budget Burn Down") if not history_df.empty else gr.update()
    bar_plot = gr.BarPlot(history_df, x="model_used", y="step", title="Model Usage Distribution", y_aggregate="count") if not history_df.empty else gr.update()
    
    return env, status, budget_txt, line_plot, bar_plot

with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🦙 Llama Inference Gateway: SLA Controller")
    env_state = gr.State(None)
    
    with gr.Row():
        diff_drop = gr.Dropdown(["easy", "medium", "hard"], value="medium", label="Traffic Scenario")
        btn_start = gr.Button("🔄 Initialize Gateway")
    
    with gr.Row():
        status_disp = gr.Markdown("Waiting to start...")
        budget_disp = gr.Markdown("Budget: $0.00")
        
    with gr.Row():
        plot_burn = gr.LinePlot()
        plot_usage = gr.BarPlot()

    gr.Markdown("### Manual Override (Route Current Prompt)")
    with gr.Row():
        btn_1b = gr.Button("Route to Llama 3.2 1B")
        btn_8b = gr.Button("Route to Llama 3.1 8B")
        btn_70b = gr.Button("Route to Llama 3.1 70B")

    btn_start.click(start_episode, inputs=[diff_drop], outputs=[env_state, status_disp, budget_disp, plot_burn, plot_usage])
    btn_1b.click(step_episode, inputs=[env_state, gr.Textbox(value="Llama 3.2 1B", visible=False)], outputs=[env_state, status_disp, budget_disp, plot_burn, plot_usage])
    btn_8b.click(step_episode, inputs=[env_state, gr.Textbox(value="Llama 3.1 8B", visible=False)], outputs=[env_state, status_disp, budget_disp, plot_burn, plot_usage])
    btn_70b.click(step_episode, inputs=[env_state, gr.Textbox(value="Llama 3.1 70B", visible=False)], outputs=[env_state, status_disp, budget_disp, plot_burn, plot_usage])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)