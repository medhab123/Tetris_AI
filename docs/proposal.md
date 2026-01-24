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
  Implement and train a baseline supervised learning model for brain metastasis segmentation using the labeled MRI scans. The model should successfully identify tumors at the same time that most professionals would, by producing segmentation masks.

- **Realistic:**  
  Explore and design a reinforcement learning agent that can learn to detect brain tumors from MRI scans, operating alongside the baseline model. The resulting system should have a measurable improvement over just the baseline.

- **Moonshot:**  
  Develop a reinforcement learning agent that can detect tumors as early as possible, either before formation or at the earliest stages; be able to detect tumors with a limited amount of data.

## AI/ML Algorithms Used

U-net convolutional neural network (CNN) (TensorFlow)

Reinforcement learning model from PyTorch

## Evaluation Plan

### Quantitative evaluation

The primary metric for evaluating our agent will be the detection success rate, which measures whether the model successfully identifies tumor regions within MRI scans. To assess the precision of these detections, we use recall and F1 score to calculate the overlaps between the predicted tumor area and ground truth segmentation; these metrics are crucial for preventing trivial solutions, such as labeling the entire image as a tumor to boost scores. Efficiency is another key focus, so we report the average number of steps or time required to identify a tumor. For cases where the agent fails, we also report the final distance to the nearest tumor to quantify partial progress and better understand the agent's behavior. To put our results in context, we use a U-Net CNN as a baseline for performance comparisons.

### Qualitative Analysis

For qualitative analysis, we will use toy examples such as a white circle on a black background to verify that the agent can reliably detect a simple target. Failure cases on real MRI scans are also examined, with attention to false positives and false negatives (e.g., checking if the agent gets confused by edema in T2 FLAIR or mistakes normal blood vessels and artifacts in T1 post-contrast images for actual tumors). As a moonshot goal, we test whether the agent can find tumors using only T1 pre-contrast images. It's a huge challenge since there's no contrast enhancement to help, but it's a great way to test the limit of what the agent can learn.

## Additional AI Tool Usage

ChatGPT was used during the project to help with brainstorming model designs and understanding difficult concepts such as reinforcement learning.

## Additional Credits

**Stanford BrainMetShare:**

This research used data provided by the Stanford Center for Artificial Intelligence in Medicine and Imaging (AIMI). AIMI curated a publicly available imaging data repository containing clinical imaging and data from Stanford Health Care, the Stanford Children's Hospital, the University Healthcare Alliance, and Packard Children's Health Alliance clinics, provisioned for research use by the Stanford Medicine Research Data Repository (STARR).
