# 🚦 LLM Gateway Router (OpenEnv)

A complete, OpenEnv-compliant reinforcement learning environment simulating an MLOps API Gateway.

## 📖 Environment Description

In production AI applications, routing every user request to a massive model (like Llama 3 70B) is too expensive, but routing everything to a small model yields poor answers for complex questions. 

**LLMRouterEnv** is an episodic environment where an AI agent acts as a dynamic router. The agent receives a stream of incoming user prompts. Its goal is to maximize the total quality of responses across the entire traffic burst without exceeding a strict rolling financial budget. The agent must learn to match prompt complexity to model capability while managing cost.

## 🕹️ Action Space

The action space is a `Discrete(3)` space representing the model routing choice:

| Action | Choice | Cost | Capability Limit |
| :--- | :--- | :--- | :--- |
| `0` | **Small Model** | $0.0001 / token | Fails on complexity > 0.3 |
| `1` | **Medium Model** | $0.0005 / token | Fails on complexity > 0.7 |
| `2` | **Large Model** | $0.0015 / token | Handles all complexity |

## 👁️ Observation Space

The observation space is a `Box(low=0.0, high=1.0, shape=(3,))` float array representing normalized telemetry data:

| Index | Name | Range | Description |
| :--- | :--- | :--- | :--- |
| `0` | `norm_length` | `[0.0, 1.0]` | The token length of the current prompt (normalized to max length). |
| `1` | `complexity` | `[0.0, 1.0]` | The semantic difficulty of the current prompt. |
| `2` | `norm_budget` | `[0.0, 1.0]` | The remaining episode budget (normalized to initial budget). |

## 🛠️ Setup Instructions

### Option A: Local Installation

1. Clone this repository and navigate to the folder.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt