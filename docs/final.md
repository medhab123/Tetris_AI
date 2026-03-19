---
layout: default
title: Final Report
---

## Project Summary

The goal of TetrisRL is to train reinforcement learning agents to play classic Atari Tetris well, ultimately in a way that is similar to how a good human player would approach the game. Tetris is actually a pretty tough environment for RL: even though the rules are simple – blocks fall, stack them, line clears – you need to think several steps ahead, understand spatial relationships, and constantly adapt to random pieces coming in. It is very different from something like Breakout or Pong, where you are mostly just reacting to what is on the screen. In Tetris, one bad placement can snowball into an unwinnable board, and it is hard to tell which earlier decision caused the problem. This phenomenon is called the temporal credit assignment problem, one of the big unsolved challenges in RL. Tetris also allows for intuitive evaluation because it is easy to tell just by watching a game whether an agent is playing smart or just randomly dropping pieces.

We built two separate RL pipelines and ran them in parallel, which also gave us a natural way to compare how the choice of algorithm and exploration strategy affects learning. The first approach uses a CNN-based agent trained with PPO (Proximal Policy Optimization) in the Gymnasium/ALE Tetris environment. The second uses a DQN (Deep Q-Network) agent, which we tested in both a custom PyGame Tetris environment we built ourselves and the ALE environment, so we could compare it directly with the PPO agent. Our main goals were: build an end-to-end training pipeline for Tetris, run experiments to examine what environment settings and training choices improve gameplay, look at reward curves and game footage to understand what the agents are doing, and explore what leads to higher and more consistent scores.

## Approach

We train and compare two distinct RL agents on Tetris. We use a controlled experimental methodology: rather than changing multiple variables at once, we establish a stable baseline for each agent and then vary one aspect at a time — environment configuration, reward shaping, or hyperparameters — comparing the resulting learning curves and episode scores to isolate what actually drives improvement.

### Agent 1: CNN + PPO (ALE Environment)

The first agent used the ALE/Tetris-v5 environment from Gymnasium, which runs the original Atari 2600 version of Tetris. There are some limitations in this environment that make learning for the agent more difficult: the agent can't see the next piece, the scoring is based on line clears rather than using the standard scoring system, and the game speed doesn't change. Furthermore, the agent cannot do moves that human players can do, such as fitting blocks into tight spaces, due to the way collision/legal block placement is defined. The agent gets raw screen pixels as input and picks from a set of discrete actions. We applied the usual Atari preprocessing steps: converting to grayscale, stacking 4 frames to enable the agent to track piece movement, skipping every 4 frames to reduce redundancy, and normalizing pixel values. The input is flattened into a vector before entering the policy network.

We used Stable Baselines3's PPO with a CNN policy. PPO is an on-policy method that clips policy updates so that training doesn't go off the rails from a single bad batch. One downside in Tetris specifically is that PPO learns a probability distribution over actions, so it doesn't account for context; the same action might be great in one board state and terrible in another. We trained for about 1 million timesteps on the HPC cluster and logged everything to TensorBoard. We ran a bunch of experiments swapping out different wrappers, normalization settings, and training lengths, always comparing back to our original baseline.

Hyperparameters (mostly Stable Baselines3 defaults):
•      Learning rate: 3 × 10⁻⁴
•      n_steps: 2048
•      Batch size: 64
•      Discount factor γ: 0.99
•      Clip range ε: 0.2
•      Total training: ~1 million timesteps

First, we recorded the agent’s performance with all default settings and no optimization, pictured below.

(insert defaults)

As expected, the agent performs incredibly poorly. The agent is unable to score any points, and at best can survive for a small period of time without dying. There was also significant fluctuation throughout the training period.

Thus, we added some environment preprocessing, such as environment wrappers and normalization. The following graphs show the performance of the agent with a vector environment, AtariWrapper, and VecFrameWrapper:

(insert defaults vs enhanced)

The vectorized environment included normalization and xxx. AtariWrapper was chosen to make the environment easier to manage for the agent, and adds. VecFrameWrapper was used to give the agent a sense of time (the agent takes in the last n images as a stack, rather than viewing each frame individually). With environment preprocessing, the agent performs much better and consistently higher than the agent without any. Thus, we used the enhanced environment to test hyperparameters on.

The hyperparameters we tested on the PPO included the following:
gamma (γ) (default = 0.99) - the variable that determines whether long-term or short-term rewards are prioritized. higher gamma prioritizes long-term rewards
clip range (default = 0.2) - the range that the policy can update within. greater clip range allows for more drastic changes
entropy coefficient (default = 0.0) - variable that tries to prevent PPO from converging too early

Of these, adjusting the entropy coefficient to 0.005 produced the most significant change from the defaults, with a slight increase in the mean reward. All other adjustments to the above hyperparameters had little effect on the mean reward in comparison to the defaults. Lowering gamma and adjusting the clip range within the range 0.1-0.3 made the average episode length more consistent than the defaults, but did not necessarily increase the length.

GAMMA:

(insert gamma images)

CLIP RANGE:

(insert clip range images)

Increasing the clip range slightly to 0.3 did the best at increasing average episode length and saw higher average peaks in terms of mean reward. Having too low of a clip range made it harder for the agent to explore new actions and update the policy in a significant manner, while having too high of a clip range resulted in much shorter episodes (faster game deaths) and very little reward gained.

ENTROPY COEFFICIENT:

(insert entropy images)

Increasing the entropy coefficient saw the most improvement in terms of mean reward, but this spike was only reached at the 1 million timestep mark. Beyond this, the drawbacks of increasing the timesteps further outweigh the possible minimal benefits of this entropy coefficient change.


### Agent 2: DQN (PyGame + ALE Environments)

## Evaluation

## Resources Used
