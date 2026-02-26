---
layout: default
title: Status
---

## Project Summary
---
Our project trains two different reinforcement learning agents to play the classic Atari Tetris. One uses the Gymnasium/ALE environment and a CNN with the PPO algorithm from Stable Baselines3, while the other uses the PyGames environment with a DQN. The agent observes the game screen as a series of images and learns through trial and error which actions (move, rotate, drop) lead to higher in‑game scores and longer survival. Our goal is to optimize each agent’s performance and have them develop strategies on par with expert Tetris play. We will do this by adjusting the training and environment parameters, such as enhancing the environment with preprocessing wrappers, how many steps we train for, and how we structure the learning process. 

## Approach
---
We started from a baseline PPO implementation that had already been set up to play Atari Tetris in the Gymnasium/ALE framework. Our first step was to clean up the training pipeline so that we could reliably launch runs on the HPC cluster, log results to TensorBoard, and inspect both learning curves and gameplay rollouts. Once we had a stable baseline model that consistently cleared at least a few lines, we began exploring optimization strategies to improve performance, such as adjusting environment wrappers, normalization settings, and the total number of training steps.

In practice, we treat the original Tetris agent as a reference model and then run controlled experiments where we change one aspect of the setup at a time (for example, how the environment is configured or how long we train) and compare the resulting learning curves and scores. 

## Evaluation
---
### Quantitative
When evaluating the agent, our initial requirements are for it to be on par with an average human player; that is, being able to achieve a reasonably good score and last for a few minutes without dying. The typical beginner player is able to score anywhere from 30,000 to 50,000 points, while a decent or somewhat regular player will score in the range of 100,000 - 300,000 points. With regards to the environment, this would be about 30-40 points for a “beginner” and 100-300 points for a “decent” player.

The next set of goalposts for the agent is achieving high level gameplay, though not necessarily equivalent to expert play. This involves achieving a score of at least 500,000 - 700,000 points, or about 500-700 points for the agent.
  
Finally, in an ideal world the agent will be able to reach the levels of experts. A pro player would be able to score close to the maximum possible score at 999,999 points, which translates to 9,999 for the ALE environment. While the current competitive play records linked underneath do include scores and playtimes beyond this, we are only testing for individual play at this time.

###Qualitative
Qualitative requirements are defined in stages. At a basic level, the agent should avoid early failure and demonstrate stable survival. At a more advanced level, it should maintain consistent board height without sudden spikes. At a higher level, the agent should minimize hole creation and preserve a clean board structure.

## Remaining Goals and Challenges
---

## Resources Used
---
Code resources:
- [ALE environment](https://ale.farama.org/environments/tetris/) to simulate Tetris
- Gymnasium documentation for [starters on agent training](https://gymnasium.farama.org/introduction/train_agent/)
- Stable Baselines 3

Determining “expert play”/”playing like a human”:
- [Tetris scoring](https://tetris.wiki/Scoring)
- [Competitive Classic Tetris Rankings](https://docs.google.com/spreadsheets/d/1NUwqOotckIdSRH2FdfhBRHPU84N6EaRBM9BDkx5nDkU/edit?gid=1685146231#gid=1685146231)
- [Common Tetris strategies](https://www.tetriseffect.game/2021/05/28/tetris-effect-community-guide/) (while they are for a slightly different version of Tetris, the core mechanics are essentially the same)

Other:
- [Reward Based Epsilon Decay](https://aakash94.github.io/Reward-Based-Epsilon-Decay/):
Reward-Based Epsilon Decay (RBED) is an exploration strategy in reinforcement learning that adjusts the ε value in ε-greedy policies based on the agent’s performance rather than time or episode count. Instead of gradually decreasing ε on a fixed schedule, RBED lowers ε only when the agent reaches a predefined reward threshold, then raises the threshold for the next stage. This creates a performance-driven transition from exploration to exploitation, ensuring that the agent reduces exploration only after demonstrating learning progress. As a result, the approach can produce more stable training, better reproducibility, and more intuitive hyperparameter tuning, although its effectiveness depends on the quality and consistency of reward signals in a given environment.
- [Introducing Q Learning](https://huggingface.co/learn/deep-rl-course/en/unit2/q-learning)
- [Batch Size for DQN](https://ai.stackexchange.com/questions/23254/is-there-a-logical-method-of-deducing-an-optimal-batch-size-when-training-a-deep)
There is no universal method to derive the optimal batch size for DQN. In practice, 32 or 64 are commonly used as default values, while larger batch sizes can be explored when aiming for the best performance. Ultimately, the optimal batch size is task-dependent and must be determined through experimentation.
- [Hidden layer](https://www.heatonresearch.com/2017/06/01/hidden-layers.html)
- [Gamma/ LR parameter settings reference](https://codesignal.com/learn/courses/q-learning-unleashed-building-intelligent-agents/lessons/introduction-to-q-learning-building-intelligent-agents)


Citations:

[ALE](https://github.com/Farama-Foundation/Arcade-Learning-Environment):

M. G. Bellemare, Y. Naddaf, J. Veness and M. Bowling. The Arcade Learning Environment: An Evaluation Platform for General Agents, Journal of Artificial Intelligence Research, Volume 47, pages 253-279, 2013.

M. C. Machado, M. G. Bellemare, E. Talvitie, J. Veness, M. J. Hausknecht, M. Bowling. Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents, Journal of Artificial Intelligence Research, Volume 61, pages 523-562, 2018

---



