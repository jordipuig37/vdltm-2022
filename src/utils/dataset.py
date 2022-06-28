from itertools import chain, repeat, tee
from more_itertools import ilen

import torch
import pytorch_lightning as pl
import pytorchvideo.data as ptv_data

from torchvision.transforms._transforms_video import (
    CenterCropVideo
)

from torchvision.transforms import Compose, Lambda

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)


class DataModule(pl.LightningDataModule):
    """This LightningDataModule implementation constructs a PyTorchVideo
    dataset for both the train validation and test partitions. It defines
    each partition's augmentation and preprocessing transforms and configures
    the PyTorch DataLoaders.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_annotation = args.train_annotation
        self.validation_annotation = args.validation_annotation
        self.test_annotation = args.test_annotation

        self.transform_params = dict(args.transform_params)
        self.clip_duration = args.clip_duration

        transform_base = [UniformTemporalSubsample(args.num_frames),
                          Lambda(lambda x: x / 255.0),
                          ShortSideScale(size=args.transform_params["side_size"]),
                          CenterCropVideo(crop_size=(args.transform_params["crop_size"],
                                          args.transform_params["crop_size"]))
        ]

        self.transform = ApplyTransformToKey(key="video",
                                             transform=Compose(transform_base))

    def train_dataloader(self):
        train_dataset = LimitDataset(self.build_base_dataset(self.train_annotation))

        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

    def val_dataloader(self):
        validation_dataset = LimitDataset(self.build_base_dataset(self.validation_annotation))

        return torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

    def test_dataloader(self):
        test_dataset = LimitDataset(self.build_base_dataset(self.test_annotation, test=True), test=True)
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

    def build_base_dataset(self, annotation_file, test=False):
        """This function builds the base for the datasets. As the only difference
        between the trhee datasets is the sampler strategy and the annotation file
        we can write this code only once :)
        """
        sampler_strategy = "uniform" if test else "random"
        return ptv_data.labeled_video_dataset(
            data_path=annotation_file,
            clip_sampler=ptv_data.make_clip_sampler(sampler_strategy, self.clip_duration),
            decode_audio=False,
            transform=self.transform,
        )


class LimitDataset(torch.utils.data.Dataset):
    """To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """
    def __init__(self, dataset, test=False):
        super().__init__()
        self.dataset = dataset
        self.test = test
        self.dataset_iter = chain.from_iterable(
            repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        if not self.test:
            # we sample one clip for each video
            return self.dataset.num_videos
        else:
            # as sample uniformly the whole video, we have more samples
            copy_to_consume, self.dataset_iter = tee(self.dataset_iter)
            return ilen(copy_to_consume)