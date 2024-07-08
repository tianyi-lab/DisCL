from torchvision import transforms


def data_loader_wrapper(config, train_file='ImageNet_LT_train', guid_values=None, ori_train=True, test=True, train=True,
                        repeat=False, limit_aug_number=False, uniform=False, merge_guid=True, augment_type='baseline',
                        downsample_ratio=1.0, txt_path=''):
    # Dataloader for ImageNet datasets, data info loaded in --datapath + '/' + dataset
    from .data_loader_ImageNet import load_data
    # Data transformation with augmentation
    data_transforms = {'train': transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), 'val': transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), 'test': transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    return load_data(config['path'], data_transforms, config, train_file=train_file, guid_values=guid_values,
                     ori_train=ori_train, test=test, train=train, repeat=repeat, limit_aug_number=limit_aug_number,
                     uniform=uniform, merge_guid=merge_guid, augment_type=augment_type,
                     downsample_ratio=downsample_ratio, txt_path=txt_path)


# Get the image normalization factor mean and std for each dataset
def get_norm_params(dataset):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return mean, std


def Custom_dataset_ImageNet(args):

    dataset = {"name": args.dataset, "class_num": 1000, "imb_factor": args.imb_factor, "path": args.datapath,
               "batch_size": 128, "sampler": None, "number_worker": 0, "pin_memory": True}
    return dataset
