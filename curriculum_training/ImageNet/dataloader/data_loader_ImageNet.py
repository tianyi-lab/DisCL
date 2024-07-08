# See https://github.com/zhmiao/OpenLongTailRecognition-OLTR/blob/master/data/dataloader.py
import os
import logging
from PIL import Image
import random
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from dataloader.sampler import get_sampler

data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                                                                       hue=0), transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'val': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'test': transforms.Compose(
                       [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


# Dataset
class LT_Dataset_MIX(Dataset):
    def __init__(self, root, txt, transform=None, guid_values=None, uniform=False, downsample_ratio=0.5,
                 augment_type='baseline'):
        # downsample_ratio: 0.29 -> 1
        self.transform = transform
        self.img_path = []
        self.targets = []
        self.guidance = []

        # only for tail classes
        dict_guid = dict()
        with open(txt, 'r') as f:
            for line in f:
                split_line = line.split()
                img_path = split_line[0]
                img_path = os.path.join(root, line.split()[0])

                label = int(split_line[1])
                if len(split_line) >= 3:
                    guidance = int(split_line[2])
                else:
                    guidance = 100

                if len(split_line) >= 4:
                    train_cnt = int(split_line[3])
                else:
                    train_cnt = 0

                # not used for current stage training
                if guid_values is not None and (guidance not in guid_values and guidance != 100):
                    continue
                if augment_type == 'baseline' and guidance != 100:
                    continue

                if train_cnt < 20:
                    # minor samples
                    if guidance not in dict_guid:
                        dict_guid[guidance] = [[], [], []]  # list of [img_path, target, guidance]

                    dict_guid[guidance][0].append(img_path)
                    dict_guid[guidance][1].append(label)
                    dict_guid[guidance][2].append(guidance)
                else:
                    # none minor samples
                    self.img_path.append(img_path)
                    self.targets.append(label)
                    self.guidance.append(guidance)

        num_nontail = len(self.img_path)
        # down-sampling the non tail classes
        down_cnt = int(downsample_ratio * num_nontail)
        list_idx = np.random.choice(num_nontail, down_cnt, replace=False)
        self.img_path = [self.img_path[i] for i in list_idx]
        self.targets = [self.targets[i] for i in list_idx]
        self.guidance = [self.guidance[i] for i in list_idx]
        logging.info(f"Downsampling over non tail data, selecting {len(list_idx)} from {num_nontail}")
        num_nontail = len(self.img_path)

        # upsampling cnt calculation
        # x/(x + y) = 0.136, x = 0.136 / (1-0.136)*y
        num_tail = int(0.136 / (1 - 0.136) * num_nontail)
        if not uniform:
            # upsampling the number of augmented data to upsampling if not -1 from dict_guid
            # current: sampling images from all guidance except original data

            if augment_type == 'sample_guid' and guid_values != [100,
                                                                 100]:  # all-level augmentation experiment / single guidance augmentation experiments
                sampling_cnt = num_tail - len(dict_guid[100][0])
                # first, adding original tail data into training data
                self.img_path.extend(dict_guid[100][0])
                self.targets.extend(dict_guid[100][1])
                self.guidance.extend(dict_guid[100][1])
                logging.info(f"Adding original tail samples into training: {len(dict_guid[100][0])}")

                # sampling images from all guidance 
                list_all_img_path = []
                list_all_targets = []
                list_all_guidance = []
                for guid, list_samples in dict_guid.items():
                    if len(dict_guid) > 1 and guid == 100:
                        continue
                    img_paths = list_samples[0]
                    labels = list_samples[1]
                    guidances = list_samples[2]
                    list_all_img_path.extend(img_paths)
                    list_all_targets.extend(labels)
                    list_all_guidance.extend(guidances)

                # random select sampling_cnt from data
                cur_samples = len(list_all_img_path)
                if cur_samples <= sampling_cnt:
                    list_idx = np.random.choice(cur_samples, sampling_cnt, replace=True)
                else:
                    list_idx = np.random.choice(cur_samples, sampling_cnt, replace=False)

                logging.info(f"Upsampling over {guid_values} data, selecting {len(list_idx)} from {cur_samples}")

                img_paths = [list_all_img_path[i] for i in list_idx]
                labels = [list_all_targets[i] for i in list_idx]
                guidances = [list_all_guidance[i] for i in list_idx]
                self.img_path.extend(img_paths)
                self.targets.extend(labels)
                self.guidance.extend(guidances)

            else:  # only augment original tail classes
                logging.info(f"Upsampling original tail samples into training")
                sampling_cnt = num_tail

                # sampling images from all guidance 
                list_all_img_path = dict_guid[100][0]
                list_all_targets = dict_guid[100][1]
                list_all_guidance = dict_guid[100][2]

                # random select sampling_cnt from data
                cur_samples = len(list_all_img_path)
                list_idx = np.random.choice(cur_samples, sampling_cnt, replace=True)
                logging.info(f"Upsampling over original data, selecting {len(list_idx)} from {cur_samples}")

                img_paths = [list_all_img_path[i] for i in list_idx]
                labels = [list_all_targets[i] for i in list_idx]
                guidances = [list_all_guidance[i] for i in list_idx]
                self.img_path.extend(img_paths)
                self.targets.extend(labels)
                self.guidance.extend(guidances)

        else:
            # random select 10% samples from data
            num_sample = 5000
            img_paths = dict_guid[100][0]
            labels = dict_guid[100][1]
            guidances = dict_guid[100][2]

            cur_samples = len(img_paths)
            list_idx = list(range(cur_samples))
            list_sel = random.choices(list_idx, k=min(cur_samples, num_sample))
            logging.info(f"uniformly sampling data, selecting {num_sample} from {cur_samples}")
            img_paths = [img_paths[i] for i in list_sel]
            labels = [labels[i] for i in list_sel]
            guidances = [guidances[i] for i in list_sel]

            self.img_path.extend(img_paths)
            self.targets.extend(labels)
            self.guidance.extend(guidances)

        num_tail = len(self.img_path) - num_nontail
        logging.info(f"Loading data {len(self.img_path)} from {txt.split('/')[-1]}")
        logging.info(
            f"Non tail samples: {num_nontail}, tail samples: {num_tail}, tail ratio: {np.round(num_tail / len(self.img_path), 4)}")  # pdb.set_trace()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.targets[index]
        guidance = self.guidance[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, guidance  # , path


# Dataset
class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None, guid_values=None, repeat=False, limit_aug_number=False, num_repeat=3,
                 uniform=False, merge_guid=True):
        self.transform = transform
        self.img_path = []
        self.targets = []
        self.guidance = []
        dict_guid = dict()
        with open(txt, 'r') as f:
            for line in f:
                split_line = line.split()
                img_path = split_line[0]
                label = int(split_line[1])
                if guid_values is not None and len(split_line) >= 3:
                    guidance = int(split_line[2])
                    if guidance not in guid_values and guidance != 100:
                        continue
                else:
                    guidance = 100

                if guidance not in dict_guid:
                    dict_guid[guidance] = [[], [], []]  # list of [img_path, target, guidance]

                img_path = os.path.join(root, line.split()[0])

                if not merge_guid and guid_values != [100, 100] and len(split_line) >= 4:
                    train_cnt = int(split_line[3])

                    if guid_values != [100] and guidance == 100 and train_cnt < 20:
                        continue

                if (repeat or guid_values == [100, 100]) and len(split_line) >= 4:
                    train_cnt = int(split_line[3])
                    if train_cnt < 20:
                        # add multiple times of minor samples into training
                        num_times = num_repeat
                        for i in range(num_times):
                            dict_guid[guidance][0].append(img_path)
                            dict_guid[guidance][1].append(label)
                            dict_guid[guidance][2].append(guidance)

                dict_guid[guidance][0].append(img_path)
                dict_guid[guidance][1].append(label)
                dict_guid[guidance][2].append(guidance)

        if not uniform:
            # limit the number of augmentation to num_repeat * 1643
            if not merge_guid:
                num_repeat += 1
            num_limits = num_repeat * 1643
            if guid_values is None:
                len_aug_guid = 0
            else:
                len_aug_guid = len(guid_values) - 1

            if len_aug_guid > 0:
                num_limits = num_limits // len_aug_guid
            for guid, list_samples in dict_guid.items():
                img_paths = list_samples[0]
                labels = list_samples[1]
                guidances = list_samples[2]

                if limit_aug_number and guid != 100:
                    cur_samples = len(img_paths)
                    list_idx = list(range(cur_samples))
                    list_sel = random.choices(list_idx, k=min(cur_samples, num_limits))
                    logging.info(f"sampling guidance data {guid}, selecting {num_limits} from {cur_samples}")
                    img_paths = [img_paths[i] for i in list_sel]
                    labels = [labels[i] for i in list_sel]
                    guidances = [guidances[i] for i in list_sel]

                self.img_path.extend(img_paths)
                self.targets.extend(labels)
                self.guidance.extend(guidances)

        else:
            # random select 10% samples from data
            num_sample = 15000
            img_paths = dict_guid[100][0]
            labels = dict_guid[100][1]
            guidances = dict_guid[100][2]

            cur_samples = len(img_paths)
            list_idx = list(range(cur_samples))
            list_sel = random.choices(list_idx, k=min(cur_samples, num_sample))
            logging.info(f"uniformly sampling data, selecting {num_sample} from {cur_samples}")
            img_paths = [img_paths[i] for i in list_sel]
            labels = [labels[i] for i in list_sel]
            guidances = [guidances[i] for i in list_sel]

            self.img_path.extend(img_paths)
            self.targets.extend(labels)
            self.guidance.extend(guidances)

        logging.info(f"Loading data {len(self.img_path)} from {txt.split('/')[-1]}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.targets[index]
        guidance = self.guidance[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, guidance  # , path


def load_data(datapath, data_transforms, params, train_file='ImageNet_LT_train', guid_values=None, ori_train=True,
              test=True, train=True, repeat=False, limit_aug_number=False, uniform=False, merge_guid=True,
              augment_type='baseline', downsample_ratio=1.0, txt_path=''):
    logging.info('Loading data from %s' % (datapath))
    dataset = params['name']
    # kwargs = {'num_workers':params['num_workers'],'pin_memory':params['pin_memory'],'drop_last':True}
    kwargs = {'num_workers': params['num_workers'], 'pin_memory': params['pin_memory']}
    train_original = f'{txt_path}/ImageNet_LT_train.txt'
    test_labelpath = f'{txt_path}/ImageNet_LT_test.txt'

    train_labelpath = f'{txt_path}/{train_file}.txt'
    test_curri_labelpath = f'{txt_path}/{train_file}_curri.txt'
    val_labelpath = f'{txt_path}/ImageNet_A_test.txt'

    # --------------------------Load Dataset from Path--------------------------#
    train_set = None
    val_set = None
    test_set = None
    dset_info = {}
    dset_info_new = {}

    if ori_train:
        train_ori_dataset = LT_Dataset(datapath, train_original, data_transforms['train'])
        # -------------------------Collect Dataset Info-----------------------------#
        num_classes = len(list(set(train_ori_dataset.targets)))
        class_loc_list = [[] for i in range(num_classes)]
        for i, label in enumerate(train_ori_dataset.targets):
            class_loc_list[label].append(i)
        img_num_per_cls = [len(x) for x in class_loc_list]
        dset_info = {'class_num': num_classes,  # 'per_class_loc': class_loc_list,
                     'per_class_img_num': img_num_per_cls}

    if train:
        if downsample_ratio != -1:
            train_dataset = LT_Dataset_MIX(datapath, train_labelpath, data_transforms['train'], guid_values=guid_values,
                                           uniform=uniform, downsample_ratio=downsample_ratio,
                                           augment_type=augment_type)

        else:
            train_dataset = LT_Dataset(datapath, train_labelpath, data_transforms['train'], guid_values=guid_values,
                                       repeat=repeat, limit_aug_number=limit_aug_number, uniform=uniform,
                                       merge_guid=merge_guid)
        # -------------------------Collect Dataset Info-----------------------------#
        num_classes = max(set(train_dataset.targets)) + 1
        class_loc_list = [[] for i in range(num_classes)]
        for i, label in enumerate(train_dataset.targets):
            class_loc_list[label].append(i)
        img_num_per_cls = [len(x) for x in class_loc_list]
        dset_info_new = {'class_num': num_classes,  # 'per_class_loc': class_loc_list,
                         'per_class_img_num': img_num_per_cls}

        # --------------------------Define Sample Strategy--------------------------#
        sampler = get_sampler(params['sampler'], train_dataset, img_num_per_cls)

        # ---------------------Create Batch Dataloader------------------------------#
        train_set = DataLoader(dataset=train_dataset, batch_size=params['batch_size'], sampler=sampler, **kwargs)

    if test:
        val_dataset = LT_Dataset(datapath, val_labelpath, data_transforms['test'])
        val_set = DataLoader(dataset=val_dataset, batch_size=params['batch_size'], shuffle=False, **kwargs)
        if not uniform:
            test_dataset = LT_Dataset(datapath, test_labelpath, data_transforms['test'])
            test_set = DataLoader(dataset=test_dataset, batch_size=params['batch_size'], shuffle=False, **kwargs)
        else:
            test_dataset = LT_Dataset(datapath, test_curri_labelpath, data_transforms['test'],
                                      guid_values=guid_values, )
            test_set = DataLoader(dataset=test_dataset, batch_size=params['batch_size'], shuffle=False, **kwargs)

    return train_set, val_set, test_set, dset_info, dset_info_new
