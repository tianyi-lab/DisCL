#!/bin/bash

SYNTHETIC_FOLDER=''
CSV_OUTPUT_FOLDER=''
PATH_CLIP_SCORE=''
PATH_TO_ORI_CSV=''
PATH_TO_WILD=''

# generate original train.csv
python datacreation_scripts/iwildcam_ori.py --save_file="${PATH_TO_ORI_CSV}" --data_dir="${PATH_TO_WILD}/train" --metadata="${PATH_TO_WILD}/metadata.csv"

# generate train.csv & curriculum validation set
python datacreation_scripts/iwildcam.py --save_folder="${CSV_OUTPUT_FOLDER}" --input_folder="${SYNTHETIC_FOLDER}" --ori_csv="${PATH_TO_ORI_CSV}" --clip_score="${PATH_CLIP_SCORE}" --curriculum

# generate train.csv
python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=256 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location="${PATH_TO_WILD}" --ft_data="${CSV_OUTPUT_FOLDER}/train.csv" --ft_data_test="${CSV_OUTPUT_FOLDER}/validate.csv" --csv-img-key filepath --csv-caption-key title --exp_name="exp_name" --curriculum --curriculum_epoch=15 --workers=4

