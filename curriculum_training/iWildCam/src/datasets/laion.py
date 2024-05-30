from dataclasses import dataclass
from multiprocessing import Value

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
import pickle

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import tokenize


def logging_input(curinput='', logger=None):
    if logger is not None:
        logger.info(curinput)
    else:
        print(curinput)
    return


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", label_key=None, guidance=None,
                 datalimit=-1, uniform_guid=False, return_guidance=False, return_img_id=False, logger=None,
                 return_train_cnt=False):
        # logging_input(f'Loading csv data from {input_filename}.', logger)
        df = pd.read_csv(input_filename, sep=sep)

        ##########################
        # mixture from original data * image guidance
        if uniform_guid:
            # only train on a uniformly distributed dataset
            if 'train_cnt' in df.columns:
                df = df[(df['guidance'] == 100) & (df['train_cnt'] <= 50)]
                df = df.sample(n=min(len(df), 10000), replace=False, ignore_index=True)
            else:
                df = df[df['guidance'] == 100]
                df = df.sample(n=min(len(df), 30000), replace=False, ignore_index=True)

            logging_input(f'sampling total data {len(df)}.', logger)

        ##########################
        # only loading guidance
        if guidance is not None and 'guidance' in df.columns:
            # only positive is included if guid != 100
            df_unenhanced = df[df['img_id'] < 0]
            df = df[(df['guidance'] == guidance) & (df['img_id'] >= 0)]
            if datalimit != -1 and len(df) > datalimit:
                df = df.sample(n=datalimit, replace=False, ignore_index=True)
                logging_input(f'sampling guid={guidance} with {len(df)} samples.', logger)
                df = pd.concat([df, df_unenhanced])

            logging_input(f'merged with unenhanced data.', logger)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        title_col = [item for item in df.columns if caption_key in item]
        num_columns = len(title_col)

        self.return_guidance = return_guidance
        if self.return_guidance:
            if 'guidance' in df.columns:
                self.guidance = df['guidance'].tolist()
            else:
                self.guidance = [100] * len(self.captions)

            if 'seed' in df.columns:
                self.seed = df['seed'].tolist()
            else:
                self.seed = [100] * len(self.captions)

        self.img_trans = T.ToPILImage()

        self.return_img_id = return_img_id
        if self.return_img_id:
            if 'img_id' in df.columns:
                self.img_id = df['img_id'].tolist()
            else:
                self.img_id = [100] * len(self.captions)

        self.captions_list = []
        for k in range(1, num_columns):
            self.captions_list.append(df[f"{caption_key}_{k}"])

        self.return_label = False
        if label_key is not None:
            self.return_label = True
            self.labels = list(map(int, df[label_key].tolist()))
            self.img_path = df["filepath"].tolist()
            self.prompt = df["title"].tolist()
        self.transforms = transforms

        self.return_train_cnt = return_train_cnt
        if self.return_train_cnt:
            if 'train_cnt' in df.columns:
                self.train_cnt = df['train_cnt'].tolist()
            else:
                self.train_cnt = [50] * len(self.captions)

        # self.classes = max(self.labels) + 1
        logging_input(f'Loading data with length {len(self.images)}.', logger)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        if img_path.endswith('.pkl'):
            with open(img_path, 'rb') as f:
                images = pickle.load(f)
            if torch.is_tensor(images):
                images = self.img_trans(images)
        else:
            images = Image.open(img_path)

        images = self.transforms(images)

        texts = tokenize([str(self.captions[idx])])[0]

        return_label = [images, texts, ]
        if len(self.captions_list) > 0:
            texts_list = [tokenize([str(self.captions_list[i][idx])])[0] for i in range(len(self.captions_list))]
            texts_list.append(texts)
            texts_list = torch.stack(texts_list, dim=0)
            perm = torch.randperm(texts_list.shape[0])
            texts_list = texts_list[perm, :]

            return_label.append(texts_list)

        if self.return_label:
            label = self.labels[idx]
            f_path = self.img_path[idx]
            f_title = self.prompt[idx]

            return_label.append(label)
            return_label.append(f_path)
            return_label.append(f_title)

        if self.return_guidance:
            guidance = self.guidance[idx]
            return_label.append(guidance)
            seed = self.seed[idx]
            return_label.append(seed)

        if self.return_img_id:
            img_id = self.img_id[idx]
            return_label.append(img_id)

        if self.return_train_cnt:
            train_cnt = self.train_cnt[idx]
            return_label.append(train_cnt)

        return return_label


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_csv_dataset(args, preprocess_fn, is_train, guidance=None, uniform_guid=False, return_guidance=False,
                    return_img_id=False, datalimit=-1, logger=None, return_train_cnt=False, progress_guid=False):
    # normal training / curriculum eval on test dataset
    input_filename = args.ft_data if is_train else args.ft_data_test
    if progress_guid:
        input_filename = args.ft_data_curri
    assert input_filename

    if args.get_labeled_csv:
        label_key = args.supervised_label_key

    else:
        label_key = None

    if not is_train:
        label_key = 'label'

    dataset = CsvDataset(input_filename, preprocess_fn, logger=logger, img_key=args.csv_img_key,
                         caption_key=args.csv_caption_key, sep=args.csv_separator, label_key=label_key,
                         guidance=guidance, datalimit=datalimit, uniform_guid=uniform_guid,
                         return_guidance=return_guidance, return_img_id=return_img_id,
                         return_train_cnt=return_train_cnt)
    num_samples = len(dataset)
    sampler = None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                            pin_memory=True, sampler=sampler, drop_last=False, )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_data(args, preprocess_fns, logger=None, guidance=None, uniform_guid=False, datalimit=-1, return_img_id=False, ):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    data["train_ft"] = get_csv_dataset(args, preprocess_train, is_train=True, guidance=guidance,
                                       uniform_guid=uniform_guid, logger=logger, datalimit=datalimit,
                                       return_img_id=return_img_id, )

    return data
