import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import logging
import argparse
from datetime import datetime

from dataloader.data_loader_wrapper import Custom_dataset_ImageNet
from utilis.config_parse import config_setup
from fine_tune_tr import fine_tune_CE


def none_or_str(value):
    if value == 'None':
        return None
    return value


parser = argparse.ArgumentParser(description='Long-Tailed Diffusion Model training')
parser.add_argument('--datapath', default='', type=str, help='dataset path')
parser.add_argument('--config', default="./config/ImageNet/ImageNet_LSC_Mixup.txt",
                    help='./config/ImageNet/ImageNet_LSC_Mixup_rn50.json, path to config file')
parser.add_argument('--train_file', default="ImageNet_LT_train", help='path to config file')
parser.add_argument('--txt_path', default="ImageNet_LT_train", help='path to config file')

parser.add_argument('--epoch', default=400, type=int, help='epoch number to train')
parser.add_argument('--dataset', default="ImageNet", type=str,
                    help='dataset name it may be CIFAR10, CIFAR100 or ImageNet')
parser.add_argument('--model_fixed', default=None, type=str, help='the encoder model path')
parser.add_argument('--checkpoint', default=None, type=str, help='model path to resume previous training, default None')
parser.add_argument('--batch_size_fc', default=128, type=int, help='CNN fully connected layer batch size')
parser.add_argument('--learning_rate_fc', default=0.001, type=float, help='CNN fully connected layer learning rate')
parser.add_argument('--exp_name', default='default', type=str, help='evaluate the model performance')
parser.add_argument('--modelStructure', default='full', type=str, help='evaluate the model performance')

parser.add_argument('--curriculum_epoch', default=60, type=int, help='evaluate the model performance')
parser.add_argument('--scheduler', default='yes', type=str, help='evaluate the model performance')
parser.add_argument('--downsample_ratio', default=1.0, type=float, help='evaluate the model performance')
parser.add_argument('--augment_type', default='sample_guid', type=str,
                    help='baseline | sample_guid, evaluate the model performance')
parser.add_argument('--loss_function', default='baseline', type=str, help='evaluate the model performance')


def main():
    args = parser.parse_args()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    saved_path = f"saved/{args.exp_name}"
    os.makedirs(saved_path, exist_ok=True)

    log_filename = args.exp_name + '_' + str(args.imb_factor) + f"_{current_time}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(os.path.join(saved_path, log_filename)), logging.StreamHandler()])
    # 1. load data
    # ouput: dataloader of ImageNet
    cfg, finish = config_setup(args.config, args.checkpoint, args.datapath, update=False)
    dataset_info = Custom_dataset_ImageNet(args)
    logging.info(f"Training file: {args.train_file}")
    logging.info(cfg)
    # fine-tuning the image net ce model
    fine_tune_CE(dataset_info, args, cfg, )

    print(" ------------Finish--------------")


if __name__ == '__main__':
    main()
