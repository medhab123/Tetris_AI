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

The vectorized environment included normalization, and runs multiple copies of the environment simultaneously to speed up training time. AtariWrapper was chosen to make the environment easier to manage for the agent, and adds. VecFrameWrapper was used to give the agent a sense of time (the agent takes in the last n images as a stack, rather than viewing each frame individually). With environment preprocessing, the agent performs much better and consistently higher than the agent without any. Thus, we used the enhanced environment to test hyperparameters on.

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

The second agent was mainly developed and tested in a custom Tetris environment we built from scratch using PyGame, following the OpenAI Gym interface. We also ran it in the ALE environment to compare directly with the PyGames version. Similar to ALE, our custom environment doesn't show the next piece and keeps game speed constant, however unlike ALE it uses the standard Tetris scoring rules (including things like level and combo bonuses).
One of the bigger design decisions for DQN was how to define the action space and state. Instead of raw pixels, each action represents a specific rotation and landing column for the current piece — basically, the agent considers every possible placement and picks whichever one has the highest predicted Q-value. Rather than feeding in pixels, we give the network four hand-crafted features describing the board after each placement.
These go into a small network with two fully connected hidden layers (64 units each, ReLU activations) that outputs a single Q-value for the placement. We used Double DQN, meaning we have a policy network that is updated at every step and a target network with frozen weights that provides more stable Q-value targets for computing the loss.

Key hyperparameters:
Batch Size: 128 - Number of samples per training update.
Learning Rate (LR): 0.0003 - Step size for updating network weights using Adam optimizer.
Discount Factor (γ): 0.99 - Controls the importance of future rewards.
Replay Buffer Size: 50,000 -Maximum number of stored transitions for experience replay.
Target Update Interval 500: Frequency of updating the target network.
Loss Function: MSE
	
### Feature

The feature vector consists of seven key components that capture both short and long term outcomes board stability. Specifically, the features include the maximum height (H), the number of lines cleared (L), the number of holes, bumpiness (B), the number of blockades (Blk), and the row and column transitions (RT and CT). Not all features are used in every model variant. However, three core features - height, holes, and line clearance are consistently included across all models, as they represent the most fundamental aspects of gameplay. The remaining features, such as bumpiness, blockages, and transitions, are selectively incorporated depending on the objective of each strategy. For instance, strategies that emphasize stability rely more on smoothness related features, while aggressive strategies may focus more on line clearing reward.

| Feature (symbol) | Description | Strategic Significance |
| ---------------- | ----------- | ---------------------- |
| Height | Aggregate height of the board | Higher stacks reduce the margin for error and increase the risk of top-out |
| Lines | Number of lines cleared by the current action | Primary source of positive reward and progress |
| Holes | Number of empty cells covered by blocks above | One of the most critical failure factors |
| Bumpiness | Sum of height differences between adjacent columns | Measures surface roughness; lower values indicate a flatter and more controllable board |
| Blockades | Number of blocks stacked above holes | Indicates how difficult it is to recover buried holes |
| Transitions (RT / CT) | Number of filled-empty transitions across rows and columns | Captures structural irregularity and board complexity |

### Reward Function

To examine whether reinforcement learning can capture different Tetris playing styles, we design multiple reward functions corresponding to distinct strategic objectives. These reward functions not only guide the agents behavior but also serve as a way to evaluate how well structured strategies can emerge from learning.

1. Survivor Mode with 7 features (linear_survivor_3)
The survivor mode agent is designed to prioritize long-term survival over score maximization. The reward function heavily penalizes board height and the number of holes, encouraging the agent to maintain a clean and low board.

$` R = 10*L *(Level + 1) + 0.5 - (10* H1.5+25*Holes+ 10 * blockades+5*B+ 2*RT+2*CT) `$

| Variable | Feature |
| -------- | ------- |
|



## Evaluation

## Resources Used
