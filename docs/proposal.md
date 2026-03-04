---
layout: default
title: Proposal
---

## Project Summary

Our project aims to create an agent that can identify brain metastasis formation as early as possible, so that it is easier to prevent fatal brain tumors in patients. Research done at Stanford has shown that the presence of brain metastases in patients is often an indicator of future brain tumor growth. Since brain metastases are much easier to treat and remove, searching for these metastases allows us to prevent them from developing further into late stage brain cancer. We will initially be working with the U-net CNN architecture, which is commonly used in medical imaging to identify tumors and other harmful developments in other parts of the body, as well as image classification in general via segmentation.

This project utilizes the Stanford Brain Metastases MRI dataset, which comprises 156 whole-brain MRI studies with multimodal imaging and tumor segmentation masks for a subset of cases. The input to the system is a multi-modal MRI scan, and the output is a tumor detection decision as well as a corresponding segmentation mask. As the baseline model, we will train a standard convolutional neural network for tumor segmentation utilizing the U-Net CNN architecture. In addition to this model, we will introduce a reinforcement learning agent that learns to sequentially explore MRI slices or regions and determine when and where to apply the segmentation model.

Instead of learning to segment tumors from scratch, the reinforcement learning agent will treat the segmentation model as a fixed tool and focus on learning an efficient search strategy. The ultimate goal of this project is to investigate whether reinforcement learning can reduce the number of slices or regions that must be examined while maintaining high tumor detection accuracy.

## Project Goals

- **Minimum:**  
 Develop an agent that can play Tetris at the level of a beginner (anywhere from 10,000 to 50,000 points for complete beginners, and 70,000 to 100,000 points for decent/casual play). Scaled to the environment, this would be equal to a score of at least 10-50 points for complete beginners and 70-100 points for casual.

- **Realistic:**  
 Develop an agent that is on par with a moderately skilled Tetris player; i.e. scores around 250-500 points in the given environment. The agent should also be able to reach this score faster and display reasonably quick decision-making skills. 

- **Moonshot:**  
  Develop an agent that can play Tetris like an expert, and develop strategies that appear in human play. The agent would be able to max out the score in the environment (9,999 for the ALE environment, for instance) as quickly as possible and in as few learning steps as possible.

## AI/ML Algorithms Used

ALE Environment: CNN with PPO algorithm

PyGame Environment: DQN, ReLu activation functions, RBED policy

## Evaluation Plan

### Quantitative evaluation

The primary metric for evaluating our agent will be the detection success rate, which measures whether the model successfully identifies tumor regions within MRI scans. To assess the precision of these detections, we use recall and F1 score to calculate the overlaps between the predicted tumor area and ground truth segmentation; these metrics are crucial for preventing trivial solutions, such as labeling the entire image as a tumor to boost scores. Efficiency is another key focus, so we report the average number of steps or time required to identify a tumor. For cases where the agent fails, we also report the final distance to the nearest tumor to quantify partial progress and better understand the agent's behavior. To put our results in context, we use a U-Net CNN as a baseline for performance comparisons.

### Qualitative Analysis

For qualitative analysis, we will use toy examples such as a white circle on a black background to verify that the agent can reliably detect a simple target. Failure cases on real MRI scans are also examined, with attention to false positives and false negatives (e.g., checking if the agent gets confused by edema in T2 FLAIR or mistakes normal blood vessels and artifacts in T1 post-contrast images for actual tumors). As a moonshot goal, we test whether the agent can find tumors using only T1 pre-contrast images. It's a huge challenge since there's no contrast enhancement to help, but it's a great way to test the limit of what the agent can learn.

## Additional AI Tool Usage

ChatGPT was used during the project to help with brainstorming, writing, and understanding the basics of difficult concepts.

## Additional Credits

[ALE](https://github.com/Farama-Foundation/Arcade-Learning-Environment):
M. G. Bellemare, Y. Naddaf, J. Veness and M. Bowling. The Arcade Learning Environment: An Evaluation Platform for General Agents, Journal of Artificial Intelligence Research, Volume 47, pages 253-279, 2013.
M. C. Machado, M. G. Bellemare, E. Talvitie, J. Veness, M. J. Hausknecht, M. Bowling. Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents, Journal of Artificial Intelligence Research, Volume 61, pages 523-562, 2018
