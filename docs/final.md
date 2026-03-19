---
layout: default
title: Final Report
---

## Project Summary

The goal of TetrisRL is to train reinforcement learning agents to play classic Atari Tetris well, ultimately in a way that is similar to how a good human player would approach the game. Tetris is actually a pretty tough environment for RL: even though the rules are simple – blocks fall, stack them, line clears – you need to think several steps ahead, understand spatial relationships, and constantly adapt to random pieces coming in. It is very different from something like Breakout or Pong, where you are mostly just reacting to what is on the screen. In Tetris, one bad placement can snowball into an unwinnable board, and it is hard to tell which earlier decision caused the problem. This phenomenon is called the temporal credit assignment problem, one of the big unsolved challenges in RL. Tetris also allows for intuitive evaluation because it is easy to tell just by watching a game whether an agent is playing smart or just randomly dropping pieces.
We built two separate RL pipelines and ran them in parallel, which also gave us a natural way to compare how the choice of algorithm and exploration strategy affects learning. The first approach uses a CNN-based agent trained with PPO (Proximal Policy Optimization) in the Gymnasium/ALE Tetris environment. The second uses a DQN (Deep Q-Network) agent, which we tested in both a custom PyGame Tetris environment we built ourselves and the ALE environment, so we could compare it directly with the PPO agent. Our main goals were: build an end-to-end training pipeline for Tetris, run experiments to examine what environment settings and training choices improve gameplay, look at reward curves and game footage to understand what the agents are doing, and explore what leads to higher and more consistent scores.

## Approach

## Evaluation

## Resources Used
