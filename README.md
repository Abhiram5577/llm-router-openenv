# 🦙 Llama Inference Gateway: SLA Controller (OpenEnv)

An OpenEnv-compliant reinforcement learning environment simulating a production-grade inference routing system for the Meta Llama ecosystem.

## 🎯 The Problem: The Over-Provisioning Trap
In production AI, routing every user prompt to `Llama 3.1 70B` ensures high quality but bankrupts the infrastructure budget. Conversely, static routing to smaller models causes unacceptable hallucination rates on complex reasoning tasks. 

**LlamaRouterEnv** frames inference routing as an episodic reinforcement learning problem. The agent acts as an API gateway, dynamically assessing prompt complexity and routing traffic across three Llama variants to maximize SLA compliance while surviving strict budget constraints.

## 🧬 Architectural Realism
Unlike static "toy" environments, this project models real-world MLOps friction:
1. **Traffic Surges:** The `hard` task simulates sudden spikes in high-complexity reasoning requests.
2. **Probabilistic Failure:** Models don't have hard cutoffs. A 1B model might occasionally succeed on a hard prompt, and a 70B model might occasionally fail, forcing the agent to learn risk management.
3. **Network Jitter:** Inference costs fluctuate slightly per step, simulating spot-instance pricing and varying token-generation lengths.

## 🕹️ Action Space
| Action | Variant | Base Cost ($) | Capability Profile |
| :--- | :--- | :--- | :--- |
| `0` | **Llama 3.2 1B** | $0.0001 / token | Optimized for trivial extraction |
| `1` | **Llama 3.1 8B** | $0.0003 / token | General purpose workhorse |
| `2` | **Llama 3.1 70B**| $0.0015 / token | Advanced reasoning and SLA guarantee |

## 🚀 Setup & Evaluation
Run the interactive telemetry dashboard to manually route traffic:
```bash
pip install -r requirements.txt
python app.py