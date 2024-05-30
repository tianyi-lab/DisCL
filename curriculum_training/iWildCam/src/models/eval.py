import torch
import numpy as np
from src.models import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.laion import get_data, get_csv_dataset

import src.datasets as datasets
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_f1_score


def logging_input(curinput='', logger=None):
    if logger is not None:
        logger.info(curinput)
    else:
        print(curinput)
    return


def process_train_stat(results, train_stats, logger, dataset_name=''):
    for key, val in results.items():
        if ('worst' in key or 'f1' in key.lower() or 'pm0' in key) and 'guidance' not in key.lower():
            logging_input(f"{dataset_name} {key}: {val:.4f}", logger)
            train_stats[dataset_name + key] = round(val, 4)
    return


def eval_single_dataset(image_classifier, dataset, args, classification_head, progress_guid=False, logger=None):

    model = image_classifier
    input_key = 'images'
    image_enc = None

    model.eval()
    classification_head.eval()

    if progress_guid:
        # run on given test set
        dataloader = get_csv_dataset(args, image_classifier.module.val_preprocess, logger=logger, is_train=False,
                                     return_guidance=True, return_img_id=True, ).dataloader


    else:
        ## equals to dataloader = dataset.test_loader
        dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=image_enc)

        # ft_iterator = iter(dataloader)  # logging_input(f'dataloader batch: {len(ft_iterator)}', logger)

    batched_data = enumerate(dataloader)
    device = args.device

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    dict_labels = dict()
    dict_preds = dict()
    dict_img_guid = dict()

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        dict_class = dict()
        dict_guidance = dict()
        # for i, data in tqdm(batched_data, total=len(batched_data)):
        for i, data in batched_data:
            # pdb.set_trace()
            data = maybe_dictionarize(data, progress_guid=progress_guid)

            x = data[input_key].to(device)
            y = data['labels'].to(device)
            if 'img_id' in data:
                img_ids = data['img_id']
            else:
                img_ids = torch.arange(i * args.batch_size, i * args.batch_size + x.shape[0])

            if 'guidance' in data:
                guidance = data['guidance']
            else:
                guidance = torch.ones_like(y)
                guidance = guidance * 100

            if 'image_paths' in data:
                image_paths = data['image_paths']

            img_emb, logits = utils.get_logits(x, model, classification_head)

            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            all_prob = F.softmax(logits, dim=-1)

            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
                correct += acc1
                n += num_total
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

                classes = torch.unique(y)
                for cls_i in classes:
                    cls_i = cls_i.item()
                    sap_ids = (y == cls_i).nonzero(as_tuple=True)
                    cur_pred = pred[sap_ids]
                    cur_correct = (cur_pred == cls_i).sum().item()
                    cur_num = len(sap_ids[0])
                    if cls_i not in dict_class:
                        dict_class[cls_i] = [0, 0]

                    dict_class[cls_i][0] += cur_correct
                    dict_class[cls_i][1] += cur_num

                if progress_guid:
                    # calculate accuracy / F1 for each guid
                    guidances = torch.unique(guidance)
                    for guid_i in guidances:
                        guid_i = guid_i.item()
                        sap_ids = (guidance == guid_i).nonzero(as_tuple=True)
                        cur_pred = pred[sap_ids]
                        cur_y = y[sap_ids]
                        # cur_guid_prob = all_prob[sap_ids]
                        # cur_probs = torch.gather(cur_guid_prob, 1, cur_y.reshape(-1, 1))

                        cur_correct = cur_pred.eq(cur_y.view_as(cur_pred)).sum().item()
                        cur_num = len(sap_ids[0])
                        if guid_i not in dict_guidance:
                            dict_guidance[guid_i] = [0, 0]

                        dict_guidance[guid_i][0] += cur_correct
                        dict_guidance[guid_i][1] += cur_num

                        if guid_i not in dict_labels:
                            dict_labels[guid_i] = []
                            dict_preds[guid_i] = []
                        dict_labels[guid_i].append(cur_y.cpu().clone().detach())
                        dict_preds[guid_i].append(cur_pred.cpu().clone().detach())

                for i, img_id_t in enumerate(img_ids):
                    img_id = img_id_t.item()
                    cur_y = y[i].item()
                    cur_prob = all_prob[i, cur_y].item()
                    cur_probs = all_prob[i].detach().cpu().numpy()
                    cur_img_emb = img_emb[i].detach().cpu().numpy()
                    cur_guid = guidance[i].item()
                    if img_id not in dict_img_guid:
                        dict_img_guid[img_id] = []
                    dict_img_guid[img_id].append([cur_y, cur_guid, cur_prob, cur_probs, cur_img_emb])

            if hasattr(dataset, 'post_loop_metrics'):
                all_labels.append(y.cpu().clone().detach())
                all_preds.append(logits.cpu().clone().detach())
                metadata = data['metadata'] if 'metadata' in data else image_paths
                all_metadata.extend(metadata)

        top1 = correct / n
        # pdb.set_trace()
        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)

            metrics = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)

            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}

    if progress_guid:
        dict_guidance_f1 = dict()
        for guid_i in dict_labels.keys():
            cur_str_labels = dict_labels[guid_i]
            cur_str_preds = dict_preds[guid_i]
            # pdb.set_trace()
            cur_str_labels = torch.cat(cur_str_labels)
            cur_str_preds = torch.cat(cur_str_preds)
            cur_str_preds = torch.squeeze(cur_str_preds)
            f1_cur_str = multiclass_f1_score(cur_str_preds, cur_str_labels, num_classes=181, average="macro")
            dict_guidance_f1[guid_i] = f1_cur_str.item()
        metrics['guidance_f1'] = dict_guidance_f1

        dict_guid_prob = dict()
        for img_id, guid_prob in dict_img_guid.items():
            for cur_guid_res in guid_prob:
                guid = cur_guid_res[1]
                prob = cur_guid_res[2]
                img_emb = cur_guid_res[-1]
                if guid not in dict_guid_prob:
                    dict_guid_prob[guid] = []
                dict_guid_prob[guid].append([prob, img_id, img_emb])

        metrics['progress_res'] = dict_guid_prob

    if 'top1' not in metrics:
        metrics['top1'] = top1

    if len(dict_class) > 0:
        metrics['class_top1'] = dict_class

    if len(dict_guidance) > 0:
        metrics['guidance_top1'] = dict_guidance

    if len(dict_img_guid) > 0:
        metrics['dict_img_guid'] = dict_img_guid

    return metrics


def evaluate(image_classifier, args, classification_head, train_stats={}, logger=None, progress_guid=False, ):
    if args.eval_datasets is None:
        return
    info = vars(args)

    if progress_guid:
        # load specific curriculum data and evaluate performance on group of guidance
        logging_input(f"Evaluating on guid level", logger)
        dataset = None

        results = eval_single_dataset(image_classifier, dataset, args, classification_head, logger=logger,
                                      progress_guid=True, )
        if 'progress_res' in results:
            dict_guid_prob = results['progress_res']
            dict_guid_prob_new = {
                key: [[item[0] for item in values], [item[1] for item in values], [item[2] for item in values]] for
                key, values in dict_guid_prob.items()}
            dict_guid_mean = {key: np.mean(values[0]) for key, values in dict_guid_prob_new.items()}
            dict_guid_std = {key: np.std(values[0]) for key, values in dict_guid_prob_new.items()}

            for key in dict_guid_mean.keys():
                guid_mean = dict_guid_mean[key]
                guid_std = dict_guid_std[key]
                # logging_input(f"Guidance = {key}: mean {np.round(guid_mean, 4)}, std {np.round(guid_std, 4)}", logger)
                train_stats[f"Guidance {key} Mean"] = guid_mean
                train_stats[f"Guidance {key} Std"] = guid_std
                train_stats[f"Guidance {key} Values"] = dict_guid_prob_new[key]

        if 'guidance_f1' in results:
            dict_guidance_f1 = results['guidance_f1']
            list_acc = [[key, value] for key, value in dict_guidance_f1.items()]

            for pair in list_acc:
                # logging_input(f"Guidance F1: {pair[0]} {pair[1]:.4f}", logger)
                train_stats[f"Guidance {pair[0]} F1"] = round(pair[1], 4)

        if 'guidance_top1' in results:
            list_acc = [[key, value[0] / value[1], value[1]] for key, value in results['guidance_top1'].items()]
            list_acc = sorted(list_acc, key=lambda x: x[1], reverse=False)
            for pair in list_acc:
                # logging_input(f"Guidance Top-1 accuracy: {pair[0]} {pair[1]:.4f}", logger)
                train_stats[f"Guidance {pair[0]} Accuracy"] = round(pair[1], 4)
                train_stats[f"Guidance {pair[0]} Number"] = pair[2]

        process_train_stat(results, train_stats, logger)

        if 'dict_img_guid' in results:
            train_stats['dict_img_guid'] = results['dict_img_guid']
        return info

    else:
        for i, dataset_name in enumerate(args.eval_datasets):
            logging_input(f"Evaluating on {dataset_name}", logger)

            dataset_class = getattr(datasets, dataset_name)
            dataset = dataset_class(image_classifier.module.val_preprocess, location=args.data_location,
                                    batch_size=args.batch_size)

            results = eval_single_dataset(image_classifier, dataset, args, classification_head)

            if 'top1' in results:
                logging_input(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}", logger)
                train_stats[dataset_name + " Accuracy"] = round(results['top1'], 4)

            if 'class_top1' in results:
                list_acc = [[key, value[0] / value[1], value[1]] for key, value in results['class_top1'].items()]
                list_acc = sorted(list_acc, key=lambda x: x[1], reverse=False)
                for pair in list_acc:
                    # logging_input(f"{dataset_name} Class Top-1 accuracy: {pair[0]} {pair[1]:.4f}", logger)
                    train_stats[dataset_name + f" Class {pair[0]} Accuracy"] = round(pair[1], 4)
                    train_stats[dataset_name + f" Class {pair[0]} Number"] = pair[2]

            process_train_stat(results, train_stats, logger, dataset_name)

        return info
