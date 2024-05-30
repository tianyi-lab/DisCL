import sys
import os

sys.path.append(os.getcwd())

from src.models.training import training
from src.models.modeling import CLIPEncoder, ImageClassifier
from src.args import parse_arguments
from logger_utils import get_logger


def main(args):

    ###logging##################################################################
    os.makedirs(args.save + args.exp_name, exist_ok=True)
    args.save = args.save + args.exp_name + "/" + "_BS" + str(args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(
        args.lr) + "_run" + str(args.run)
    os.makedirs("expt_logs/" + args.exp_name, exist_ok=True)
    logging_path = "expt_logs/" + args.exp_name + "/" + "_BS" + str(args.batch_size) + "_WD" + str(
        args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
    os.makedirs(logging_path, exist_ok=True)

    log_filename = logging_path + "/log.log"
    logger = get_logger(l_name='DisCL Logger', l_file=log_filename)
    assert args.save is not None, 'Please provide a path to store models'
    #############################################################################

    # Initialize the CLIP encoder
    clip_encoder = CLIPEncoder(args, keep_lang=True)
    logger.info(args)
    finetuned_checkpoint = training(args, clip_encoder, logger)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
