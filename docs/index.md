# TetrisRL

TetrisRL is our CS 175 project on teaching an AI agent to play classic Atari Tetris. We use reinforcement learning (PPO + deep neural networks) so the agent learns strategies like stacking and line clears directly from gameplay experience, without any hard‑coded rules. Our goal is to understand what training and environment choices actually help the agent reach higher scores and more stable gameplay.

---

## Project Overview

This project trains and evaluates a reinforcement learning agent in the Gymnasium/ALE Tetris environment. The agent observes the game screen, chooses joystick actions (move, rotate, drop), and receives points for clearing lines. Over time it improves by trial and error as we experiment with different optimization strategies (e.g., environment wrappers, normalization, and training length) to see which setups lead to better Tetris play.

**Key goals:**

- Build a reliable end‑to‑end RL training pipeline for Tetris  
- Optimize the agent using different training/environment strategies  
- Analyze learning behavior with TensorBoard curves and gameplay rollouts  
- Compare different model variants to understand what leads to high scores  

---

## Demo / Screenshots

![Training screenshot](screenshots.png)

We visualize training progress and gameplay in TensorBoard using reward and episode‑length curves, along with recorded policy rollouts that show how the agent’s play style changes over time.

---

## Links

- **Source code repo:** https://github.com/abby-liao/tetrisRL 
- **ALE Tetris environment docs:** https://ale.farama.org/environments/tetris  
- **Gymnasium RL docs:** https://gymnasium.farama.org/introduction/train_agent/  

---

## Reports

- [Proposal](proposal.html)
- [Status](status.html)
- [Final](final.html)
