#!/bin/bash

SYNTHETIC_FOLDER=''
CSV_OUTPUT_FOLDER=''
PATH_CLIP_SCORE=''
PATH_TO_INLT=''
PATH_TO_TXT=''  # LT_train.txt / test.txt


# generate train.csv
python datacreation_scripts/imagenet_LT.py --save_file="${CSV_OUTPUT_FOLDER}" --data_dir="${PATH_TO_INLT}/train" --input_folder="${SYNTHETIC_FOLDER}" --clip_score="${PATH_CLIP_SCORE}"

# generate test.csv
python datacreation_scripts/imagenet_LT.py --save_file="${CSV_OUTPUT_FOLDER}" --data_dir="${PATH_TO_INLT}/test" --test --input_folder="${SYNTHETIC_FOLDER}"

modelver="full"
downsample_ratio=0.3

python3 main.py --epoch=65 --train_file="ImageNet_LT_train_aug" --exp_name="aug_${modelver}_ds" --modelStructure=${modelver} --slurm_job_id=$SLURM_JOB_ID --curriculum_epoch=60  --learning_rate_fc=0.001 --scheduler="yes" --augment_type="sample_guid" --downsample_ratio=${downsample_ratio} --txt_path="${PATH_TO_TXT}"

