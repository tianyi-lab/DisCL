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

### iWildCam

1. Prepare for a data csv file including hard samples
    - Template of the csv file is shown in file [sample.csv](data_generation/iWildCam/sample.csv)
2. Use this csv to generate synthetic data with guidance scales & random seeds
    ```shell
    python3 data_generation/iWildCam/gene_img.py --part=1 --total_parts=1 --data_csv="${PATH_TO_CSV}" --output_path="${OUTPUT_FOLDER}"
    ```
3. Compute CLIPScore for filtering out poor-quality images.
    ```shell
    python3 data_generation/iWildCam/comp_clip_scores.py --syn_path="${OUTPUT_FOLDER}" --real_path="${PATH_TO_WILDS}"
    ```
    - Results 1 (clip_score.pkl): including the image-image similarity score and image-text similarity score
    - Results 2 (filtered_results.pkl): including only the filtered image-image similarity score and image-text
      similarity score

### ImageNet-LT

1. Prepare for a data csv file including hard samples
    - Template of the csv file is shown in file [sample.csv](data_generation/ImageNet_LT/sample.csv)
2. Use this csv to generate diversified text prompt for hard classes
    ```shell
    python3 data_generation/ImageNet_LT/get_text_prompt.py --data_csv="${PATH_TO_CSV}" --prompt_json="${PATH_TO_PROMPT}" 
    ```
3. Use this csv to generate synthetic data with guidance scales & random seeds
    ```shell
    python3 data_generation/ImageNet_LT/gene_img.py --part=1 --total_parts=1 --data_csv="${PATH_TO_CSV}" --output_path="${OUTPUT_FOLDER}" --prompt_json="${PATH_TO_PROMPT}" 
    ```
4. Compute CLIPScore for filtering out poor-quality images. This script will produce a clip_score.pkl including the
   image-image similarity score and image-text similarity score
    ```shell
    python3 data_generation/ImageNet_LT/comp_clip_scores.py --syn_path="${OUTPUT_FOLDER}" --real_path="${PATH_TO_INLT}"
    ```
    - Results 1 (clip_score.pkl): including the image-image similarity score and image-text similarity score
    - Results 2 (filtered_results.pkl): including only the filtered image-image similarity score and image-text
      similarity score

## Code for Curriculum Learning

### For ImageNet-LT

### For iWildCam
