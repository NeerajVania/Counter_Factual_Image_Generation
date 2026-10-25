import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from torchvision import datasets
import random

from .torchsampler import ImbalancedDatasetSampler
from .utils import make_balanced_val_split

from sklearn.model_selection import train_test_split, GroupShuffleSplit

class EPtransform(object):
    def __init__(self, image_size) -> None:
        self.image_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(image_size, antialias=True),
                v2.Normalize(
                    mean=[0.3704248070716858, 0.2282254546880722, 0.13915641605854034],
                    std=[0.23381589353084564, 0.1512117236852646, 0.09653093665838242]
                )
            ]
        )
    def __call__(self, image):
        return self.image_transform(image)
    

class ConditionalRotation:
    def __init__(self, degrees, tolerance=0.05):
        self.degrees = degrees
        self.tolerance = tolerance

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image_tensor = v2.functional.to_tensor(image)
        else:
            image_tensor = image

        top_row = image_tensor[:, 0, :]
        bottom_row = image_tensor[:, -1, :]

        top_nonblack_frac = (top_row > 0).float().mean().item()
        bottom_nonblack_frac = (bottom_row > 0).float().mean().item()

        if top_nonblack_frac > self.tolerance and bottom_nonblack_frac > self.tolerance:
            angle = random.uniform(-self.degrees, self.degrees)
            image = v2.functional.rotate(image, angle)
        return image
    
class DiffusionTransform(object):
    def __init__(self, image_size) -> None:
        self.image_transform = v2.Compose(
            [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            ConditionalRotation(degrees=15),
            v2.Resize(image_size, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.01, hue=0.01),
            v2.Normalize(
                mean=[0.3704248070716858, 0.2282254546880722, 0.13915641605854034],
                std=[0.23381589353084564, 0.1512117236852646, 0.09653093665838242]
            )
            ]
        )
    def __call__(self, image):
        return self.image_transform(image)
    
class TrainTransform(object):
    def __init__(self, image_size) -> None:
        self.image_transform = v2.Compose(
            [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(image_size, scale=(0.4, 1.15), ratio=(0.7, 1.3)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(180),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.Normalize(
                mean=[0.3704248070716858, 0.2282254546880722, 0.13915641605854034],
                std=[0.23381589353084564, 0.1512117236852646, 0.09653093665838242]
            )
            ]
        )
    def __call__(self, image):
        return self.image_transform(image)

class EncoderTransform(object):
    def __init__(self, image_size) -> None:
        self.image_transform = v2.Compose(
            [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomResizedCrop(image_size, scale=(0.87, 1.15), ratio=(0.7, 1.3)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(180),
            v2.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.01, hue=0.01),
            v2.Normalize(
                mean=[0.3704248070716858, 0.2282254546880722, 0.13915641605854034],
                std=[0.23381589353084564, 0.1512117236852646, 0.09653093665838242]
            )
            ]
        )
    def __call__(self, image):
        return self.image_transform(image), self.image_transform(image)


class EyePacsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, cfg, use_train_transform=False, use_diffusion_transform=False, use_encoder_transform=False):
        self.data_frame = pd.read_csv(csv_file, low_memory=False)
        self.data_frame = self.data_frame[
            ~self.data_frame['session_image_quality'].isin(['Insufficient for Full Interpretation'])
        ].reset_index(drop=True)
        self.data_frame = self.data_frame[
            self.data_frame['diagnosis_image_dr_level'].notna() & self.data_frame['image_path'].notna()
        ].reset_index(drop=True)
        self.data_frame = self.data_frame[self.data_frame['confidence']>0.2].reset_index(drop=True)
        self._iq = self.data_frame['confidence']
        self._labels = self.data_frame['diagnosis_image_dr_level']
        self._labels = torch.tensor(self._labels.to_list(), dtype=torch.long)
        self.root_dir = root_dir
        self.image_transform = EPtransform((cfg.data.image_resolution, cfg.data.image_resolution))
        self.use_train_transform = use_train_transform
        self.train_transform = TrainTransform((cfg.data.image_resolution, cfg.data.image_resolution))
        self.use_diffusion_transform = use_diffusion_transform
        self.diffusion_transform = DiffusionTransform((cfg.data.image_resolution, cfg.data.image_resolution))
        self.use_encoder_transform = use_encoder_transform
        self.encoder_transform = EncoderTransform((cfg.data.image_resolution, cfg.data.image_resolution))

    def get_labels(self):
        return self._labels.tolist()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image_path = self.root_dir + self.data_frame.loc[idx, 'image_path']
        image = Image.open(image_path)
        if self.use_train_transform:
            image = self.train_transform(image)
        elif self.use_diffusion_transform:
            image = self.diffusion_transform(image)
        elif self.use_encoder_transform:
            image1, image2 = self.encoder_transform(image)
            return image1, image2, self._labels[idx]
        else:
            image = self.image_transform(image)
        if self.data_frame.loc[idx, 'image_side'] == 'left':
            image = v2.functional.hflip(image)
        label = self._labels[idx]
        iq = self._iq[idx]
        return image, label, iq

def get_dataloader(cfg, mode="train"):
    """Returns the dataloader."""

    """
    # Some example augmentations applicable to ImageNet

    image_resolution = cfg.data.image_resolution

    rotation_degrees_max = cfg.data.augmentation.rotation_degrees_max
    scale_range = cfg.data.augmentation.scale_range
    aspect_ratio_factor = cfg.data.augmentation.aspect_ratio_factor
    
    crop_resolution = int(0.5+np.sqrt(2)*image_resolution)
    scale_min = 2**(-scale_range/2)
    scale_max = 2**(scale_range/2)
    aspect_ratio_min = 1/aspect_ratio_factor
    aspect_ratio_max = aspect_ratio_factor
    
    transforms = [
        v2.ToImage(),
        v2.RandomResizedCrop(
            size=(crop_resolution, crop_resolution),
            scale=(scale_min, scale_max),
            ratio=(aspect_ratio_min, aspect_ratio_max),
            antialias=True,
            interpolation=v2.InterpolationMode.BILINEAR,
        ),
        v2.RandomRotation(
            degrees=rotation_degrees_max,
            interpolation=v2.InterpolationMode.BILINEAR,
        ),
        v2.CenterCrop(image_resolution),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.GaussianNoise(sigma=0.01),
    ] if mode == 'train' else [
        v2.ToImage(),
        v2.Resize(image_resolution),
        v2.CenterCrop(image_resolution),
        v2.ToDtype(torch.float32, scale=True),
    ]
    """
    diffusion_transform = cfg.data.diffusion_transform

    dataset = EyePacsDataset(
        csv_file=cfg.data.csv_file,
        root_dir=cfg.data.root_dir,
        cfg=cfg,
        use_diffusion_transform=diffusion_transform,
        use_encoder_transform=cfg.data.encoder_transform
        )
    
    if cfg.model.name == "VAE" or cfg.model.name == "encoder":
        return DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    print("Using imbalanced sampler:", cfg.data.use_imbalanced_sampler)
    print("Using Diffusion Transform:", diffusion_transform)

    groups = dataset.data_frame['patient_id'].values

    # trainval_idx covers the full dataset (no held-out test split for DiT/RF)
    import numpy as np
    trainval_idx = np.arange(len(dataset))
    trainval_groups = groups[trainval_idx]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=cfg.seed)

    train_subidx, val_subidx = next(gss.split(trainval_idx, groups=trainval_groups))

    train_idcs = trainval_idx[train_subidx]
    val_idcs = trainval_idx[val_subidx]

    train_dataset = Subset(dataset, train_idcs)
    val_dataset = Subset(dataset, val_idcs)

    sampler = ImbalancedDatasetSampler(train_dataset) if cfg.data.use_imbalanced_sampler else None

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=True, sampler=sampler, shuffle=(sampler is None))
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True)

    if cfg.model.name != "resnet":
        return train_dataloader if mode == "train" else val_dataloader

    if cfg.model.name == "resnet":
        dataset = EyePacsDataset(
            csv_file=cfg.data.csv_file,
            root_dir=cfg.data.root_dir,
            cfg=cfg,
            use_train_transform=False
        )

        groups = dataset.data_frame['patient_id'].values

        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=cfg.seed)

        trainval_idx, test_idcs = next(gss.split(range(len(dataset)), groups=groups))

        trainval_groups = groups[trainval_idx]

        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=cfg.seed)

        train_subidx, val_subidx = next(gss_val.split(trainval_idx, groups=trainval_groups))

        train_idcs = trainval_idx[train_subidx]
        val_idcs = trainval_idx[val_subidx]

        train_dataset = Subset(
            EyePacsDataset(
                csv_file=cfg.data.csv_file,
                root_dir=cfg.data.root_dir,
                cfg=cfg,
                use_train_transform=True
            ),
            train_idcs
        )
        val_dataset = Subset(dataset, val_idcs)
        test_dataset = Subset(dataset, test_idcs)


        sampler = ImbalancedDatasetSampler(train_dataset) if cfg.data.use_imbalanced_sampler else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )

        print(f"Train size: {len(train_dataset)}")
        print(f"Val size: {len(val_dataset)}")
        print(f"Test size: {len(test_dataset)}")

        print("Unique patients:")
        print("Train:", len(set(groups[train_idcs])))
        print("Val:", len(set(groups[val_idcs])))
        print("Test:", len(set(groups[test_idcs])))

        print("Overlap of patients:")
        print("Train & Val:", len(set(groups[train_idcs]) & set(groups[val_idcs])))
        print("Train & Test:", len(set(groups[train_idcs]) & set(groups[test_idcs])))
        print("Val & Test:", len(set(groups[val_idcs]) & set(groups[test_idcs])))


    if mode == "train":
        return train_loader
    elif mode == "val":
        return val_loader
    elif mode == "test":
        return test_loader