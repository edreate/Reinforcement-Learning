# [**Edreate.com – Deep Reinforcement Learning Course**](https://edreate.com/courses/deep-reinforcement-learning/)

Welcome 👋  
This repository contains the **codebase** used in lessons from Edreate’s Deep Reinforcement Learning (DRL) course.

<p align="center">
  <a href="https://edreate.com/courses/deep-reinforcement-learning/">
    <img src="https://raw.githubusercontent.com/edreate/Brand-Identity-Media/main/Logo/RGB/Logo/SVG/EdReate_Logo.svg" alt="EdReate Logo" width="200"/>
  </a>
</p>

👉 For the **full learning experience**—including in-depth write-ups, mathematical formulas, video explanations, and structured chapters—visit the course page:  
🔗 [edreate.com/courses/deep-reinforcement-learning](https://edreate.com/courses/deep-reinforcement-learning/)

---

## 🤝 Community

Join our [Discord server](https://discord.gg/KUstJ2jf) for learning, collaboration, and Q&A.

---

## 🚀 Setup Instructions

For complete setup details, see:  
[Setting Up Coding Environment and Dependencies](https://edreate.com/courses/deep-reinforcement-learning/setting-up-for-rl-course/setting-up-coding-environment-and-dependencies/)

### Quickstart (TL;DR)
You’ll need **Python** and **uv** installed.

```bash
# install uv (if not already installed)
pip install uv
```
```bash
# install all dependencies into .venv
uv sync
```

```bash
# activate the virtual environment
source .venv/bin/activate
```

```bash
# launch Jupyter
uv run jupyter notebook
```

💡 You can also use your favorite code editor (VS Code, PyCharm, etc.).

---

## 🌟 Algorithms Covered (Course Highlights)

<p align="center">
  <b>This repository tracks the main algorithms from the Deep RL course.</b><br/>
  Completed ones link to full lessons, others are marked <i>Coming Soon!</i>
</p>

### ✅ Available Now
- [**Tabular Q-Learning**](https://github.com/edreate/Reinforcement-Learning/tree/main/src/q-learning) – start with the introductory notebook and walk through a simple 2×3 grid world, then try the stochastic/complex variant to stress-test your policy updates (`src/q-learning/q_learning.ipynb`, `q_learning_2x3_simple_world.ipynb`).
- [**Deep Q-Learning**](https://github.com/edreate/Deep_Reinforcement_Learning/blob/main/01_deep_q_learning.ipynb)
  Learn how DQN scales beyond Q-tables and train agents directly with neural networks.

### 🔜 Coming Soon
- **Vanilla Policy Gradient (VPG)** – direct optimization of stochastic policies  
- **Actor–Critic (A2C)** – combining value functions with policy learning  
- **Proximal Policy Optimization (PPO)** – stable, scalable policy gradients  
- **Advanced Methods** – SAC and more

<p align="center">
  🚧 More lessons and code will be added as the course grows!
</p>
    
---

## 🏁 Benchmark & Use Trained Policies

- **Benchmark yourself**: run the interactive human baseline for Lunar Lander and see how your manual rewards compare.
  ```bash
  uv run python src/human-benchmark/00_human_lunar_lander_benchmark.py
  ```
  Use the arrow keys to control thrust and record your scores across episodes.

- **Fly trained agents**: plug your saved weights into the Lunar Lander viewers in `src/run-lunar-lander/`.
  - PyTorch: point `MODEL_FILE_PATH` in `LunarLander_in_Action_PyTorch.py` to your checkpoint (discrete or continuous) and run:
    ```bash
    uv run python src/run-lunar-lander/LunarLander_in_Action_PyTorch.py
    ```
  - ONNX: export your policy and update the ONNX path in `LunarLander_in_Action_ONNX.py`, then launch:
    ```bash
    uv run python src/run-lunar-lander/LunarLander_in_Action_ONNX.py
    ```

---

## 📄 License

This project is licensed under the terms of the  
[LICENSE](./LICENSE) file in the root of this repository.
