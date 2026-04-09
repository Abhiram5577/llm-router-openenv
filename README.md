---
title: Intelligent LLM Routing Gateway
emoji: 🦙
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# 🦙 Intelligent LLM Routing Gateway: SLA Controller

An enterprise-ready reinforcement learning environment and dynamic API middleware simulating a production-grade inference routing system for the Meta Llama ecosystem. This project solves the dynamic cost-vs-quality routing problem in modern MLOps.

## 🎯 The Problem: The Over-Provisioning Trap
In production AI, routing every user prompt to `Llama 3.1 70B` ensures high quality but bankrupts the infrastructure budget. Conversely, static routing to smaller models causes unacceptable hallucination rates on complex reasoning tasks. 

This project frames inference routing as an episodic reinforcement learning problem. The agent acts as an API gateway, dynamically assessing prompt complexity and routing traffic across three Llama variants to maximize SLA compliance while surviving strict budget constraints.

## 🧬 Architectural Realism
Unlike static RL environments, this project models real-world MLOps friction:
1. **Traffic Surges:** Simulates sudden spikes in high-complexity reasoning requests.
2. **Probabilistic Failure:** Models don't have hard cutoffs. A 1B model might occasionally succeed on a hard prompt, and a 70B model might occasionally fail, forcing the agent to learn risk management.
3. **Network Jitter:** Inference costs fluctuate slightly per step, simulating spot-instance pricing and varying token-generation lengths.

## 🕹️ Action Space
The environment routes traffic to one of three dynamic endpoints:

| Action | Variant | Base Cost ($) | Capability Profile |
| :--- | :--- | :--- | :--- |
| `0` | **Llama 3.2 1B** | $0.0001 / token | Optimized for trivial extraction & fast responses |
| `1` | **Llama 3.1 8B** | $0.0003 / token | General purpose workhorse |
| `2` | **Llama 3.1 70B**| $0.0015 / token | Advanced reasoning and SLA guarantee |

## 🏗️ Core Infrastructure & Safeguards
This gateway implements several enterprise-level infrastructure safeguards for secure, scalable deployment:
* **Multi-Mode Deployment:** Seamlessly operates as both a persistent web service (FastAPI on port `7860`) and a standalone background inference engine.
* **Dynamic Credential Management:** Loads proxy endpoints and credentials strictly via injected environment variables (`API_BASE_URL`, `API_KEY`), ensuring zero hardcoded secrets.
* **Network Resilience:** Implements robust `try/except` output parsing and API timeout safeguards (`timeout=30.0`) to prevent hung connections and graceful degradation.
* **Modern Dependency Locking:** Built using the `uv` package manager with strict lockfile (`uv.lock`) compliance for highly reproducible Docker deployments.

## 🚀 Setup & Deployment

This project is containerized for Docker/Hugging Face Spaces but can be run locally using the `uv` package manager.

### Prerequisites
* Python 3.10+
* `uv` package manager (`pip install uv`)
* An active Hugging Face inference token

### 1. Web Server Mode (Daemonized)
Starts the FastAPI readiness probes (`/health`, `/reset`) and safely launches the background inference daemon.
```bash
# Install dependencies and sync lockfile
uv pip install -r requirements.txt

# Run as a Python module
python -m server.app
