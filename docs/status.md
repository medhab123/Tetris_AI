---
layout: default
title: Status
---

## Project Summary
---
Our project trains a reinforcement learning agent to play Atari Tetris using the Gymnasium/ALE environment and the PPO algorithm from Stable Baselines3. The agent observes the game screen and learns through trial and error which actions (move, rotate, drop) lead to higher in‑game scores and longer survival. Our overall goal is to build a reliable end‑to‑end training pipeline and then optimize the agent’s performance using different training and environment strategies, such as adjusting how the environment is wrapped, how many steps we train for, and how we structure the learning process. By systematically trying these strategies and analyzing learning curves and gameplay videos, we aim to better understand what makes reinforcement learning agents succeed or fail on a challenging, classic game like Tetris.

## Approach
---
We started from a baseline PPO implementation that had already been set up to play Atari Tetris in the Gymnasium/ALE framework. Our first step was to clean up the training pipeline so that we could reliably launch runs on the HPC cluster, log results to TensorBoard, and inspect both learning curves and gameplay rollouts. Once we had a stable baseline model that consistently cleared at least a few lines, we began exploring optimization strategies to improve performance, such as adjusting environment wrappers, normalization settings, and the total number of training steps.

In practice, we treat the original Tetris agent as a reference model and then run controlled experiments where we change one aspect of the setup at a time (for example, how the environment is configured or how long we train) and compare the resulting learning curves and scores. 

### Evaluation
---

## Remaining Goals and Challenges
---

## Resources Used

Code resources:
- [ALE environment](https://ale.farama.org/environments/tetris/) to simulate Tetris
- Gymnasium documentation for [starters on agent training](https://gymnasium.farama.org/introduction/train_agent/)
- Stable Baselines 3

Determining “expert play”/”playing like a human”:
- [Tetris scoring](https://tetris.wiki/Scoring)
- [Competitive Classic Tetris Rankings](https://docs.google.com/spreadsheets/d/1NUwqOotckIdSRH2FdfhBRHPU84N6EaRBM9BDkx5nDkU/edit?gid=1685146231#gid=1685146231)
- [Common Tetris strategies](https://www.tetriseffect.game/2021/05/28/tetris-effect-community-guide/) (while they are for a slightly different version of Tetris, the core mechanics are essentially the same)
- 
Citations:
[ALE](https://github.com/Farama-Foundation/Arcade-Learning-Environment):
M. G. Bellemare, Y. Naddaf, J. Veness and M. Bowling. The Arcade Learning Environment: An Evaluation Platform for General Agents, Journal of Artificial Intelligence Research, Volume 47, pages 253-279, 2013.

M. C. Machado, M. G. Bellemare, E. Talvitie, J. Veness, M. J. Hausknecht, M. Bowling. Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents, Journal of Artificial Intelligence Research, Volume 61, pages 523-562, 2018

---



