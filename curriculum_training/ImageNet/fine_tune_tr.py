import logging

import torch
from model.metrics import *
from dataloader.data_loader_wrapper import data_loader_wrapper
from torch.optim.lr_scheduler import CosineAnnealingLR

from utilis.test_ft import test_ft
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import os
import pickle
from typing import List, Dict, Tuple


def evaluate_loop(args, dataloader, model, loss_fn, device, dset_info, epoch, train_dset_info, ):
    # Get number of batches
    num_batches = len(dataloader)

    test_loss, correct, total = 0, 0, 0

    probs, preds, labels, guids = [], [], [], []

    # since we dont need to update the gradients, we use torch.no_grad()
    with torch.no_grad():
        for data in dataloader:
            # Every data instance is an image + label pair
            img, label, guid = data
            # Transfer data to target device
            img = img.to(device)
            label = label.to(device)
            labels.append(label)

            # Compute prediction for this batch
            logit = model(img)

            # compute the loss
            test_loss += loss_fn(logit, label).item()

            # Calculate the index of maximum logit as the predicted label
            prob = F.softmax(logit, dim=1)
            probs.extend(list(prob.squeeze().cpu().numpy()))
            pred = prob.argmax(dim=1)
            preds.extend(list(pred.cpu().numpy()))
            guids.extend(list(guid.cpu().numpy()))

            # record correct predictions
            correct += (pred == label).type(torch.float).sum().item()
            total += label.size(0)

    # -----------------Post Compensation Accuracy-------------------------------#
    probs = np.array(probs)
    labels = torch.cat(labels)
    preds = np.array(preds)
    guids = np.array(guids)

    # -----------------save pickles for analysis-------------------------------#
    saved_path = f"saved/{args.exp_name}"
    os.makedirs(saved_path, exist_ok=True)
    with open(f"{saved_path}/predict_{epoch}.pkl", 'wb') as f:
        pickle.dump((probs, labels, preds, dset_info['per_class_img_num']), f)

    _, mmf_acc = get_metrics(probs, labels, dset_info['per_class_img_num'], )
    # Gather data and report
    test_loss /= num_batches
    accuracy = correct / total
    logging.info("Test Error:   Accuracy: {:.2f}, Avg loss: {:.4f} ".format(100 * accuracy, test_loss))
    print("Test Error:   Accuracy: {:.2f}, Avg loss: {:.4f} ".format(100 * accuracy, test_loss))

    label_shift_acc = 0
    mmf_acc_pc = None

    logging.info("\n")
    print("\n\n")
    return test_loss, accuracy, label_shift_acc, mmf_acc, mmf_acc_pc, (probs, labels, preds, guids)


def get_metrics(probs, labels, cls_num_list, ):
    labels = [tensor.cpu().item() for tensor in labels]
    acc = acc_cal(probs, labels, method='top1')

    mmf_acc = list(mmf_acc_cal(probs, labels, cls_num_list))
    logging.info('Many Medium Few shot Top1 Acc: ' + str(mmf_acc))
    print('Many Medium Few shot Top1 Acc: ' + str(mmf_acc))

    return acc, mmf_acc


def find_curri_type(args, epoch, ):
    ctype = 'all_aug'
    if args.curriculum_strategy == 'none':
        # no curriculum strategy
        # using all data for training
        ctype = 'all_aug'
    elif args.curriculum_strategy == 'all_shift':
        # no curriculum strategy
        # using all data for training
        ctype = 'all_shift'
    elif args.curriculum_strategy == 'repeat':
        # repeat minor class
        ctype = 'repeat'
    elif args.curriculum_strategy == 'shrink':
        ctype = 'shrink'
    else:
        # with the curriculum strategy
        if args.curriculum_epoch is not None and epoch > args.curriculum_epoch:
            # outside the curriculum epochs
            ctype = 'out_curri'
        else:
            # still within the curriculum epoch
            if 'fixed' in args.curriculum_strategy:
                ctype = 'in_curri_fixed'
            elif args.curriculum_strategy == 'recursive':
                # recursively iterate over guidance
                ctype = 'in_curri_recur'
            elif args.curriculum_strategy == 'progress':
                ctype = 'progress_guid'
    return ctype


def seq_curri_guid(args, list_guidance: List, cur_guidance_id=None, cur_str_times=1, list_loop=1, epoch=0,
                   direction='increase'):
    ctype = find_curri_type(args, epoch)
    if ctype == 'all_aug':
        # simplily using all data we have
        if args.guidance is not None:
            if args.merge_guid == '100':
                guid_values = [args.guidance, 100]
            else:
                guid_values = [args.guidance, ]
        else:
            guid_values = list_guidance  # guid_values = None
        cur_guid_id = 0

    elif ctype == 'all_shift':
        # simplily using all data we have
        if epoch < args.curriculum_epoch:
            guid_values = list_guidance  # guid_values = None
        else:
            cur_str_times = 1
            cur_guidance_id = list_guidance.index(100)
            guid_values = [100, 100]
        cur_guid_id = 0

    elif ctype == 'repeat':
        guid_values = [100]
        cur_guid_id = 0

    elif ctype == 'shrink':
        # fixed sequence of guidance
        if cur_str_times < [cur_guidance_id]:
            # cur_guidance_id unchanged 
            cur_str_times += 1
        else:
            # remove guidance from data
            cur_str_times = 1
            cur_guidance_id += 1

            if cur_guidance_id >= len(list_guidance):
                cur_guidance_id = len(list_guidance) - 1

        guid_values = list_guidance[cur_guidance_id:]

    elif ctype == 'in_curri_recur':
        cur_guidance = list_guidance[cur_guidance_id]
        if args.merge_guid == '100':
            guid_values = [cur_guidance, 100]
        else:
            guid_values = [cur_guidance, ]

        # recursively iterate over guidance
        if cur_guidance_id == len(list_guidance) - 1:
            # already the largest id
            direction = 'decrease'
            cur_guidance_id -= 1
        elif cur_guidance_id == 0:
            # already the smallest id
            direction = 'increase'
            cur_guidance_id += 1
        else:
            # in process
            if direction == 'increase':
                cur_guidance_id += 1
            else:
                cur_guidance_id -= 1

    elif ctype == 'in_curri_fixed':
        # fixed sequence of guidance
        if cur_str_times < list_loop[cur_guidance_id]:
            # cur_guidance_id unchanged 
            cur_str_times += 1
        else:
            cur_str_times = 1
            cur_guidance_id += 1

            if cur_guidance_id >= len(list_guidance):
                cur_guidance_id = len(list_guidance) - 1

        cur_guidance = list_guidance[cur_guidance_id]
        if args.merge_guid == '100':
            guid_values = [cur_guidance, 100]
        else:
            guid_values = [cur_guidance, ]

    elif ctype == 'out_curri':
        cur_str_times = 1
        cur_guidance_id = list_guidance.index(100)
        guid_values = [100, 100]

    elif ctype == 'progress_guid':
        # guid_values = list_guidance
        guid_values = None

    else:
        raise ValueError(f"invalid ctype {ctype}")

    logging.info(f"Selecting guidances: {guid_values} for epoch {epoch}")
    return cur_guidance_id, guid_values, cur_str_times, ctype, direction


def process_results(eval_res):
    probs, labels, preds, guids = eval_res
    dict_guid_prob = dict()
    for i in range(len(guids)):
        cur_guid = guids[i].item()
        cur_label = labels[i].item()
        cur_probs = probs[i, cur_label].item()
        if cur_guid not in dict_guid_prob:
            dict_guid_prob[cur_guid] = []
        dict_guid_prob[cur_guid].append(cur_probs)
    return dict_guid_prob


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.learning_rate_fc * epoch / 5
    elif epoch > 80:
        lr = args.learning_rate_fc * 0.01
    elif epoch > 60:
        lr = args.learning_rate_fc * 0.1
    else:
        lr = args.learning_rate_fc

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def fine_tune_CE(dataset_info, args, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_fn = nn.CrossEntropyLoss()

    # load the model for fine tuning
    model = test_ft(datapath=dataset_info["path"], args=args, modelpath=args.model_fixed, crt_modelpath=None,
                    test_cfg=None)
    model.to(device)

    # First, freeze all layers and unfreeze the fully connected layer
    if args.modelStructure == 'onlyfc':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    # loading training set used for all epoch
    # no curriculum settings
    logging.info(f"Training file: {args.train_file}")
    candidate_guid = [10, 30, 50, 100]

    logging.info(f"Candidate Guid: {candidate_guid}")
    merge_guid = True

    train_set, val_set, test_set, dset_info, _ = data_loader_wrapper(cfg.dataset, train_file=args.train_file,
                                                                     guid_values=candidate_guid, ori_train=True,
                                                                     test=True, repeat=False, limit_aug_number=True,
                                                                     merge_guid=merge_guid,
                                                                     augment_type=args.augment_type,
                                                                     downsample_ratio=args.downsample_ratio,
                                                                     txt_path=args.txt_path)

    best_accuracy, best_label_shift_acc = 0.0, 0.0
    best_mmf_acc_ce, best_mmf_acc_pc = [], []
    cur_guidance_id = 0
    cur_str_times = 1

    if args.curriculum_epoch is not None:
        # find looping times based on curriculum epoch
        loop_times = args.curriculum_epoch // len(candidate_guid)
    else:
        # find looping times based on entire epoch
        loop_times = args.epoch // len(candidate_guid)

    list_loop = [loop_times] * len(candidate_guid)
    if sum(list_loop) < args.curriculum_epoch:
        epoch_diff = args.curriculum_epoch - sum(list_loop)
        # add to early stage
        for i in range(epoch_diff):
            list_loop[i] += 1
        logging.info(f"loop times for each guidance: {list_loop}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate_fc, weight_decay=0.00001)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_set) * args.epoch, eta_min=0, last_epoch=-1, verbose=False)
    direction = 'increase'
    for epoch in range(args.epoch):
        model.train()
        running_loss = 0.0
        total_loss = 0.0
        correct, total = 0, 0

        cur_guidance_id, guid_values, cur_str_times, c_type, direction = seq_curri_guid(args=args, epoch=epoch,
                                                                                        cur_guidance_id=cur_guidance_id,
                                                                                        list_guidance=candidate_guid,
                                                                                        list_loop=list_loop,
                                                                                        cur_str_times=cur_str_times,
                                                                                        direction=direction)

        if c_type == 'repeat':
            repeat = True
        else:
            repeat = False

        if c_type in ('all_aug', 'all_shift'):
            limit_aug_number = True
        else:
            limit_aug_number = True

        if c_type == 'progress_guid':
            # select guidance values based on progress
            # first run on uniform dataset
            # evaluate the progress of samples with different guidance scale
            # decide the final guidance scale

            # 1. evaluate the probability of samples with different guid scale
            output = data_loader_wrapper(cfg.dataset, train_file=args.train_file, guid_values=candidate_guid,
                                         ori_train=False, train=False, uniform=True, txt_path=args.txt_path)
            progress_eval = output[2]

            output = evaluate_loop(args, progress_eval, model, loss_fn, device, dset_info, epoch=epoch,
                                   train_dset_info=dset_info, )
            dict_guid_prob = process_results(output[-1])

            # 2. train on uniform set
            loader_out = data_loader_wrapper(cfg.dataset, train_file=args.train_file, guid_values=candidate_guid,
                                             ori_train=False, uniform=True, txt_path=args.txt_path)

            train_set, _, _, _, cur_dset_info = data_loader_wrapper(cfg.dataset, train_file=args.train_file,
                                                                    guid_values=guid_values, ori_train=False,
                                                                    test=False, repeat=repeat, uniform=True,
                                                                    limit_aug_number=limit_aug_number,
                                                                    merge_guid=merge_guid,
                                                                    augment_type=args.augment_type,
                                                                    downsample_ratio=args.downsample_ratio,
                                                                    txt_path=args.txt_path)

            uniform_set = loader_out[0]
            print(f"number of uniform batch: {len(uniform_set)}")
            logging.info(f"number of uniform batch: {len(uniform_set)}")

            for batch, data in enumerate(uniform_set):
                image, labels, _ = data
                image, labels = image.to(device), labels.to(device)

                optimizer.zero_grad()

                logits = model(image)
                probs = F.softmax(logits, dim=1)
                prediction = probs.argmax(dim=1)

                # record correct predictions
                correct += (prediction == labels).type(torch.float).sum().item()
                total += labels.size(0)

                # compute the loss and its gradients
                loss = loss_fn(logits, labels)
                # Backpropagation
                loss.backward()

                # update the parameters according to gradients
                optimizer.step()
                if args.scheduler == 'yes':
                    scheduler.step()

            # 3. evaluate probability again
            new_output = evaluate_loop(args, progress_eval, model, loss_fn, device, dset_info, epoch=epoch,
                                       train_dset_info=dset_info)
            dict_guid_prob_new = process_results(new_output[-1])

            dict_guid_diff = dict()
            list_guid_mean = []
            for guid, new_prob in dict_guid_prob_new.items():
                prev_prob = dict_guid_prob[guid]
                prob_diff = np.array(new_prob) - np.array(prev_prob)
                dict_guid_diff[guid] = prob_diff
                diff_mean = np.mean(prob_diff)
                diff_std = np.std(prob_diff)
                list_guid_mean.append([guid, diff_mean, diff_std])
                logging.info(f"Guid {guid}: mean {np.round(diff_mean, 4)}, std: {np.round(diff_std, 4)}")

            # 4. find the best guidance scale
            list_guid_mean = sorted(list_guid_mean, key=lambda x: x[1], reverse=True)

            logging.info(f"Selecting guidance {list_guid_mean[0][0]}")
            best_guid = [list_guid_mean[0][0]]
            if args.merge_guid == '100':
                best_guid.append(100)
            logging.info(f"Selecting guidance {best_guid}")
            train_set, _, _, _, cur_dset_info = data_loader_wrapper(cfg.dataset, train_file=args.train_file,
                                                                    guid_values=guid_values, ori_train=False,
                                                                    test=False, repeat=repeat,
                                                                    limit_aug_number=limit_aug_number,
                                                                    merge_guid=merge_guid,
                                                                    augment_type=args.augment_type,
                                                                    downsample_ratio=args.downsample_ratio,
                                                                    txt_path=args.txt_path)


        else:
            train_set, _, _, _, cur_dset_info = data_loader_wrapper(cfg.dataset, train_file=args.train_file,
                                                                    guid_values=guid_values, ori_train=False,
                                                                    test=False, repeat=repeat,
                                                                    limit_aug_number=limit_aug_number,
                                                                    merge_guid=merge_guid,
                                                                    augment_type=args.augment_type,
                                                                    downsample_ratio=args.downsample_ratio,
                                                                    txt_path=args.txt_path)

        print(f"number of batch: {len(train_set)}")
        logging.info(f"number of batch: {len(train_set)}")

        for batch, data in enumerate(train_set):
            image, labels, _ = data
            image, labels = image.to(device), labels.to(device)

            logits = model(image)
            probs = F.softmax(logits, dim=1)
            prediction = probs.argmax(dim=1)

            # record correct predictions
            correct += (prediction == labels).type(torch.float).sum().item()
            total += labels.size(0)

            # compute the loss and its gradients
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()

            # Backpropagation
            loss.backward()

            # update the parameters according to gradients
            optimizer.step()
            if args.scheduler == 'yes':
                scheduler.step()
                lr = scheduler.get_lr()[0]

            # Gather data and report
            running_loss += loss.item()
            total_loss += loss.item()

        cnn_running_loss, train_accuray = total_loss / (batch + 1), correct / total
        logging.info(
            "cnn training loss is {}; cnn training accuracy is {:.2f}".format(cnn_running_loss, train_accuray * 100))
        print("cnn training loss is {}; cnn training accuracy is {:.2f}".format(cnn_running_loss, train_accuray * 100))

        # evaluate the model
        model.eval()
        print("epoch: {}: ".format(epoch))
        logging.info("epoch: {}: ".format(epoch))
        valid_loss, valid_accuracy, label_shift_acc, mmf_acc, mmf_acc_pc, _ = evaluate_loop(args, test_set, model,
                                                                                            loss_fn, device, dset_info,
                                                                                            epoch,
                                                                                            train_dset_info=cur_dset_info)
        _ = evaluate_loop(args, val_set, model, loss_fn, device, dset_info, epoch=epoch, train_dset_info=dset_info, )

        # --------------------------Save state_dicts------------------------#
        if not isinstance(model, nn.DataParallel):
            cfg.update(['state_dict', 'model'], model.state_dict())
        else:
            cfg.update(['state_dict', 'model'], model.module.state_dict())
        cfg.update(['state_dict', 'optimizer'], optimizer.state_dict())

        if valid_accuracy > best_accuracy:  # save the model with best validation accuracy
            best_accuracy = valid_accuracy
            best_mmf_acc_ce = mmf_acc

            folder_path = f"checkpoint/{args.exp_name}"
            os.makedirs(folder_path, exist_ok=True)
            save_path_ce = f"{folder_path}/ckpt_best_ce.checkpoint"

            cfg.save(save_path_ce)

        if label_shift_acc > best_label_shift_acc:
            best_label_shift_acc = label_shift_acc
            best_mmf_acc_pc = mmf_acc_pc

            folder_path = f"/{args.exp_name}"
            os.makedirs(folder_path, exist_ok=True)
            save_path_pc = f"{folder_path}/ckpt_best_pc.checkpoint"

            cfg.save(save_path_pc)

    logging.info(
        " The best accuracy is {}, The best label shift accuracy is {}".format(best_accuracy, best_label_shift_acc))
    logging.info(" the best accuracy ce mmf is {}; the best acc pc mmf is {}".format(best_mmf_acc_ce, best_mmf_acc_pc))
    print(" The best accuracy is {}, The best label shift accuracy is {}".format(best_accuracy, best_label_shift_acc))
