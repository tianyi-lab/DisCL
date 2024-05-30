import os
import copy
from tqdm import trange, tqdm

import torch
import pandas as pd
from clip.loss import ClipLoss

from typing import List
from src.models.eval import evaluate
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, get_logits
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets.laion import get_data
import random
import math
import numpy as np


def set_seed(seed: int = 42, if_torch: bool = True) -> None:
    """
    Set random
    """
    np.random.seed(seed)
    random.seed(seed)
    if if_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


set_seed(0)


def seq_curri_guid(list_guidance: List, cur_guidance_id=None, cur_str_times=None, ctype='out_curri', loop_times=1):
    # sequentially use guidance 
    if ctype == 'no_curri':
        # iteratively loop over all guidance
        cur_guidance_id += 1
        if cur_guidance_id >= len(list_guidance):
            cur_guidance_id = 0  # guidance = 0
        cur_guidance = list_guidance[cur_guidance_id]
        return cur_guidance_id, cur_guidance

    elif ctype == 'in_curri':
        # have fixed curriculum length
        if cur_str_times < loop_times:
            # cur_guidance_id unchanged 
            cur_str_times += 1
        else:
            cur_str_times = 1
            cur_guidance_id += 1

            if cur_guidance_id >= len(list_guidance):
                cur_guidance_id = len(list_guidance) - 1

        cur_guidance = list_guidance[cur_guidance_id]
        return cur_guidance_id, cur_guidance, cur_str_times

    elif ctype == 'out_curri':
        cur_guidance = 100
        cur_str_times = 1
        cur_guidance_id = list_guidance.index(cur_guidance)
        return cur_guidance_id, cur_guidance, cur_str_times
    else:
        raise ValueError(f"invalid ctype {ctype}")


def load_data(logger, args, clip_encoder, cur_guidance=None, cur_str_times=1, epoch=0, uniform_guid=False, ):
    if cur_guidance is not None:
        logger.info(f"loading image guidance = {cur_guidance}, loop times {cur_str_times}")

    # load dataloader
    img_text_data = get_data(args, (clip_encoder.train_preprocess, clip_encoder.val_preprocess), epoch=0,
                             return_img_id=True, datalimit=args.datalimit, guidance=cur_guidance,
                             uniform_guid=uniform_guid, logger=logger)
    assert len(img_text_data), 'At least one train or eval dataset must be specified.'

    ft_dataloader = img_text_data['train_ft'].dataloader
    return ft_dataloader


def generate_class_head(model, args, epoch):
    # get classification head embedding
    args.current_epoch = epoch
    classification_head_new = get_zeroshot_classifier(args, model.module.model)
    classification_head_new = classification_head_new.cuda()
    return classification_head_new


def general_eval(model, args, stats, epoch: int, logger, print_log=False, log_dir=None, ):
    """

    :param model:
    :param args:
    :param stats:
    :param epoch:
    :param logger:
    :param print_log:
    :param log_dir:
    :return:
    """

    epoch_stats = {}
    epoch_stats['Epoch'] = epoch
    epoch_stats['epoch'] = epoch
    classification_head_new = generate_class_head(model, args, epoch)
    _ = evaluate(model, args, classification_head_new, epoch_stats, logger=logger)

    ood_acc = 0
    num_datasets = 0
    for k, v in epoch_stats.items():
        if 'Accuracy' in k:
            ood_acc += v
            num_datasets += 1
    if num_datasets != 0:
        ood_acc = ood_acc / num_datasets
    else:
        ood_acc = 0

    epoch_stats['Avg OOD Acc'] = round(ood_acc, 4)
    if print_log:
        logger.info(f"Avg OOD Acc : {ood_acc:.4f}")

    epoch_stats = {key: values for key, values in epoch_stats.items() if ' Class' not in key}

    if log_dir is not None:
        if 'dict_img_guid' in epoch_stats:
            del epoch_stats['dict_img_guid']
        stats.append(epoch_stats)
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(log_dir + '/stats.tsv', sep='\t')

    return stats


def progress_eval(model, args, last_perform, epoch: int, logger, progress_guid=False, print_log=True, ):
    """
    Find best guidance based on guid group
    :param print_log:
    :param model:
    :param args:
    :param last_perform:
    :param epoch:
    :param logger:
    :param progress_guid:
    :return:
    """

    def remove_outliers(data):
        # Calculate Q1, Q3, and IQR
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        # Determine outliers using IQR
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out outliers
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        return filtered_data

    def rnd_prog(input_progress, ):
        return np.round(input_progress, 6)

    classification_head_new = generate_class_head(model, args, epoch)
    Dict_cur_guidance = {}

    _ = evaluate(model, args, classification_head_new, train_stats=Dict_cur_guidance, logger=logger,
                 progress_guid=progress_guid)
    str_progress = dict()
    res_progress = dict()
    saved_diff = dict()

    keywords = 'Values'
    logger.info(f"Computing progress based on metric {keywords}")

    dict_guid_prog = {}
    for key, value in Dict_cur_guidance.items():
        if 'Number' in key:
            continue
        if keywords not in key:
            continue

        if key not in last_perform:
            if isinstance(value, float):
                last_perform[key] = 0
            else:
                last_perform[key] = copy.deepcopy(value)
                last_perform[key][0] = np.zeros_like(last_perform[key][0])

        list_img_id = copy.deepcopy(value[1])
        list_img_emb = copy.deepcopy(value[2])
        value = value[0]

        # guidance value
        list_replace = ['Strength', 'Guidance', ' Accuracy', ' F1', ' Values']
        guidance_i = copy.deepcopy(key)
        for replace_word in list_replace:
            guidance_i = guidance_i.replace(replace_word, '')
        guidance_i = int(guidance_i)

        # progress as relative increase of prob in each image
        value_arr = np.array(value)
        last_arr = np.array(last_perform[key][:2])[0, :]
        cur_progress = value_arr - last_arr
        saved_diff[guidance_i] = [value_arr.copy(), last_arr.copy(), list_img_id, list_img_emb]  # saved for

        # remove outliers
        cur_progress = remove_outliers(cur_progress)
        dict_guid_prog[guidance_i] = cur_progress

        # use 75% quantile as criteria
        thres_diff = np.percentile(cur_progress, 75)

        # relative_diff = cur_progress / value_arr
        mean_diff = np.mean(cur_progress)
        std_diff = np.std(cur_progress)

        str_progress[f"Guidance {guidance_i}"] = rnd_prog(mean_diff)  # for logging
        # res_progress[guidance_i] = np.max(cur_progress) - np.min(cur_progress)  # for guidance ranking
        # res_progress[guidance_i] = std_diff  # for guidance ranking
        res_progress[guidance_i] = mean_diff  # for guidance ranking
        if print_log:
            logger.info(
                f"Guidance {guidance_i}, 75%: {rnd_prog(thres_diff)}, mean: {rnd_prog(mean_diff)}, std: {rnd_prog(std_diff)}")

    last_perform = copy.deepcopy(Dict_cur_guidance)
    list_sample_prob = []

    return res_progress, str_progress, last_perform, saved_diff, list_sample_prob


def init_guidance_setting(args, ):

    df_ori = pd.read_csv(args.ft_data, delimiter='\t')

    len_data = len(df_ori)
    list_guidance = list(set(df_ori['guidance'].values.tolist()))
    list_guidance = sorted(list_guidance, reverse=False)  # 0 --> 100
    # list_guidance = sorted(list_guidance, reverse=True)  # 100 --> 0

    # using curriculum_epoch to decide the current guidance
    # finish viewing all guidance data during curriculum_epoch
    len_ori = len(df_ori[df_ori['guidance'] == 100])
    num_batch_ori = int(len_ori / args.batch_size)  # num of batch in non curriculum epoch (update iterations)

    # estimate the number of times loading for each guid
    len_all_guid = len(df_ori[df_ori['guidance'] != 100])
    num_guid = len(list_guidance)
    loop_times = int(args.curriculum_epoch / num_guid)

    # start from guidance = 100
    cur_guidance_id = 0
    cur_guidance = list_guidance[cur_guidance_id]

    return cur_guidance_id, cur_guidance, list_guidance, loop_times, len_data, num_batch_ori


def training(args, clip_encoder, logger):
    model_path = ''

    assert args.train_dataset is not None, "Please provide a training dataset."
    logger.info('Fine-tuning Using FLYP Loss')
    model = clip_encoder
    clip_encoder.process_images = True
    print_every = 100
    clip_loss_fn = ClipLoss(local_loss=False, gather_with_grad=False, cache_labels=True, rank=0, world_size=1,
                            use_horovod=False)

    log_dir = "expt_logs/" + args.exp_name + "/" + "_BS" + str(args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(
        args.lr) + "_run" + str(args.run)
    os.makedirs(log_dir, exist_ok=True)

    devices = list(range(torch.cuda.device_count()))
    logger.info('Using devices' + str(devices))

    model = model.cuda()

    ############################
    # Data initialization
    cur_str_times = 1
    start_epoch = -1
    load_ckpt = False

    logger.info(f"Training dataset {args.train_dataset}")

    model = torch.nn.DataParallel(model, device_ids=devices)

    model.train()

    clip_params = list(model.parameters())
    total_params = clip_params
    params = [p for p in total_params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    init_data = init_guidance_setting(args, )
    cur_guidance_id, cur_guidance, list_guidance, loop_times, len_data, num_batch_ori = init_data

    step = 0
    stats = []
    last_perform = {}

    ############################
    # load data
    ft_dataloader = load_data(logger, args, clip_encoder, cur_guidance=cur_guidance, cur_str_times=cur_str_times,
                              epoch=0, )
    ft_iterator = iter(ft_dataloader)
    num_batches = len(ft_dataloader)
    if args.curriculum:
        if args.curriculum_epoch is None:
            num_batches = int(len_data / args.batch_size) if len_data is not None else num_batches * len(list_guidance)
        else:
            num_batches = num_batch_ori
    logger.info(f"Num batches is {num_batches}")

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, (args.epochs + 1) * num_batches, args.min_lr)

    total_iter = 0
    pre_guidance = None
    start_uniform = 0
    next_change_guid = False

    if args.uniform_set:
        start_uniform = total_iter
        # start with guid found on uniformly distributed dataset
        eval_res = progress_eval(model, args, last_perform, 0, logger, progress_guid=True, print_log=False)
        last_perform = eval_res[2]

        ft_dataloader = load_data(logger, args, clip_encoder, epoch=0, uniform_guid=True)
        next_change_guid = True
        ft_iterator = iter(ft_dataloader)

    # record the progress history (prob diff)
    # when compute the current progress, progress = 0.8 * current + 0.2 * previous progress
    adjust_epoch = False
    for epoch in trange(start_epoch + 1, args.epochs):
        # If set curriculum epochs
        if args.curriculum_epoch is not None and epoch >= args.curriculum_epoch:
            if cur_guidance != 100:
                logger.info('Restart dataloader')
                cur_guidance = 100

                ft_dataloader = load_data(logger, args, clip_encoder, cur_guidance=cur_guidance, epoch=epoch, )
                ft_iterator = iter(ft_dataloader)
                num_batches = len(ft_dataloader)

        logger.info(f"Epoch : {epoch}")
        epoch_stats = {}
        epoch_stats['Epoch'] = epoch
        epoch_stats['epoch'] = epoch

        id_flyp_loss_sum = 0
        model.train()
        model = model.cuda()

        for i in trange(num_batches):

            step += 1
            optimizer.zero_grad()
            if load_ckpt and step >= num_batches * args.curriculum_epoch and not adjust_epoch:
                # adjust the gap of steps of two experiments
                adjust_epoch = True
                break

            try:
                ft_batch = next(ft_iterator)
            except StopIteration:
                uniform_set = False  # run on uniform set right not
                skip_loading = False
                if epoch > args.curriculum_epoch:
                    # train on baseline without curriculum strategy / curriculum period ends
                    skip_loading = True

                elif args.progress_guid:
                    # select next guid based on progress
                    if args.uniform_set and not next_change_guid:
                        # not training progress eval to find the best guid
                        # run training on uniformly distributed dataset first
                        # evaluate the improvement on this uniformly distributed dataset
                        # use the largest improvement as the next guid
                        logger.info(f"Running on uniform set")
                        cur_guidance = None
                        uniform_set = True
                        next_change_guid = True
                        start_uniform = total_iter

                    else:
                        next_change_guid = False

                        if args.random_guid:
                            # not running progress eval
                            cur_guidance = random.choice(list_guidance)
                            logger.info(f"randomly select guid {cur_guidance}")
                            cur_str_times = 0
                        else:
                            # find the largest guidance based on progress
                            eval_res = progress_eval(model, args, last_perform, epoch, logger, progress_guid=True, )
                            res_progress, _, last_perform, saved_diff, _ = eval_res

                            list_progress = [(guid, prog) for guid, prog in res_progress.items()]
                            list_progress = sorted(list_progress, key=lambda x: x[-1], reverse=True)
                            largest_guid = list_progress[0]

                            next_guid = largest_guid
                            logger.info(f"Select largest guid = {next_guid[0]}")

                            cur_guidance = next_guid[0]
                            cur_str_times = 0

                if not skip_loading:
                    if not args.progress_sample or uniform_set:
                        ft_dataloader = load_data(logger, args, clip_encoder, cur_guidance=cur_guidance,
                                                  cur_str_times=cur_str_times, epoch=epoch, uniform_guid=uniform_set, )
                    else:
                        # select training samples
                        ft_dataloader = load_data(logger, args, clip_encoder, epoch=epoch, )

                ft_iterator = iter(ft_dataloader)
                ft_batch = next(ft_iterator)

            ft_image, ft_text, ft_imgid = ft_batch

            ft_image, ft_text = ft_image.cuda(), ft_text.cuda()
            ft_image_features, ft_text_features, logit_scale2 = model(ft_image, ft_text)
            if len(logit_scale2.shape) >= 1:
                logit_scale2 = logit_scale2[0]
            ft_clip_loss_peritem = clip_loss_fn(ft_image_features, ft_text_features, logit_scale2)

            ft_clip_loss = torch.mean(ft_clip_loss_peritem)
            ft_clip_loss.backward()
            optimizer.step()
            scheduler(step)

            id_flyp_loss_sum += ft_clip_loss.item()

            if i % print_every == 0:
                percent_complete = 100 * i / num_batches
                logger.info(f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"
                            f"Loss: {ft_clip_loss.item():.4f}")

            if args.uniform_set and total_iter - start_uniform == 1:
                if args.progress_guid:
                    # start with guid found on uniformly distributed dataset
                    eval_res = progress_eval(model, args, last_perform, epoch, logger, progress_guid=True,
                                             print_log=False, )
                    last_perform = eval_res[2]

            total_iter += 1

        #############################################
        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch}.pt')
            torch.save({'model_state_dict': model.module.state_dict(), }, model_path)
            logger.info('Saving model to' + str(model_path))

        #############################################
        # Evaluate
        logger.info(f"Formal evaluation ...")
        stats = general_eval(model, args, stats, epoch, logger=logger, print_log=True, log_dir=log_dir)

    if args.save is not None:
        return model_path
