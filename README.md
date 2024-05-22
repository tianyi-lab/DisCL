# DisCL

[Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion]() (Will be released in Arxiv soon)

CREDITS: Our code is heavily based
on [FLYP](https://github.com/locuslab/FLYP), [LDMLR](https://github.com/AlvinHan123/LDMLR/tree/main),
and [Open CLIP](https://github.com/mlfoundations/open_clip). We greatly thank the authors for open sourcing their code!

## Overview

<p align="center">
  <img width="750" src="assets/overview.png"> 
</p>

Our approach is composed of two phases: (Phase 1) Interpolated Synthetic Generation and (Phase 2) Training with CL. In
Phase 1, we use a model pretrained on the original data to identify the ''hard'' samples, and generate data with a full
spectrum from synthetic to real images with various image guidance $\lambda$. In Phase 2, utilizing this full spectrum
of data, we design a curriculum strategy (Non-Adaptive or Adaptive), depending on the task. According to the selected
strategy, image guidance is selected at each training stage. Synthetic data generated with selected guidance is then
combined with real data of non-hard samples for training task-model.

## Installation

```shell
conda create -n DisCL python=3.10
conda activate DisCL
pip3 install open_clip_torch
pip3 install wilds
pip3 install -r requirements.txt
```

## Dataset

We use two public datasets for training : ImageNet-LT and iWildCam.

- ImageNet-LT is a long-tailed subset of [ImageNet](https://image-net.org/download.php) data. Long-tailed meta
  information could be download
  from [google drive](https://drive.google.com/drive/folders/19cl6GK5B3p5CxzVBy5i4cWSmBy9-rT_-).
- iWildCam is a image classification dataset captured by wildlife camera trap. It is release by WILDS and can be
  downloaded with its offical [package](https://github.com/p-lambda/wilds/tree/main).

## Code for Synthetic Data Generation

## Code for Curriculum Learning

### For ImageNet-LT

### For iWildCam
