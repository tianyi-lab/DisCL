#!/bin/bash

PATH_TO_CSV=''
PATH_TO_PROMPT=''
OUTPUT_FOLDER=''
PATH_TO_INLT=''

# Prepare for a csv file containing difficult images


# Generate text prompt for ImageNet-LT Classes
python3 data_generation/ImageNet_LT/get_text_prompt.py --prompt_json="${PATH_TO_PROMPT}" --data_csv="${PATH_TO_CSV}"

# Generate images using Stable Diffusion model
python3 data_generation/ImageNet_LT/gene_img.py --part=1 --total_parts=1 --prompt_json="${PATH_TO_PROMPT}" --data_csv="${PATH_TO_CSV}" --output_path="${OUTPUT_FOLDER}"

# Generate CLIPScore for syn & real, syn & text
python3 data_generation/ImageNet_LT/comp_clip_scores.py --syn_path="${OUTPUT_FOLDER}" --real_path="${PATH_TO_INLT}"
