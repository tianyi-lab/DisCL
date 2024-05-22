#!/bin/bash

PATH_TO_CSV=''
OUTPUT_FOLDER=''
PATH_TO_WILDS=''

# Prepare for a csv file containing difficult images


# Generate images using Stable Diffusion model
python3 data_generation/iWildCam/gene_img.py --part=1 --total_parts=1 --data_csv="${PATH_TO_CSV}" --output_path="${OUTPUT_FOLDER}"

# Generate CLIPScore for syn & real, syn & text
python3 data_generation/iWildCam/comp_clip_scores.py --syn_path="${OUTPUT_FOLDER}" --real_path="${PATH_TO_WILDS}"
