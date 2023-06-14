import os
import random
from abc import abstractmethod
from typing import Any

import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BaseDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        metadata_filename: str,
        random_seed: int,
        transforms: transforms.Compose,
    ) -> None:
        self.metadata_filepath = os.path.join(root_dir, metadata_filename)
        self.videos_dir = os.path.join(root_dir, 'videos')
        self.metadata = self._load_metadata()
        self.label_2_idx = self._create_label_mapping()
        random.seed(random_seed)
        self.transforms = transforms

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        ...

    @property
    def num_classes(self) -> int:
        return len(self.label_2_idx)

    def _load_metadata(self) -> pd.DataFrame:
        return pd.read_csv(self.metadata_filepath)

    def _create_label_mapping(self) -> dict[str, int]:
        unique_labels = self.metadata['label'].unique()
        label_mapping = {}
        for idx, label in enumerate(sorted(unique_labels)):
            label_mapping[label] = idx
        return label_mapping

    def _load_video(self, video_filepath: str) -> torch.Tensor:
        return torchvision.io.read_video(
            video_filepath, output_format='TCHW',
        )[0] / 255

    def _get_video_filepath(self, idx: int) -> str:
        video_filename = self.metadata['video_id'].iloc[idx]
        return os.path.join(self.videos_dir, video_filename)

    def _get_label(self, idx: int) -> torch.Tensor:
        label_string = self.metadata['label'].iloc[idx]
        label = torch.zeros(self.num_classes)
        label[self.label_2_idx[label_string]] = 1
        return label


class FrameDataset(BaseDataset):
    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_filepath = self._get_video_filepath(idx)
        video = self._load_video(video_filepath)
        frame_idx = random.randint(0, video.size(0) - 1)
        label = self._get_label(idx)
        return self.transforms(video[frame_idx]), label


class VideoDataset(BaseDataset):
    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_filepath = self._get_video_filepath(idx)
        video = self._load_video(video_filepath)
        label = self._get_label(idx)
        video = torch.stack([self.transforms(frame) for frame in video])
        return video, label


def get_transforms(
    resize: int,
    crop: int,
    mean: list[float],
    std: list[float],
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(resize, antialias=False),
            transforms.CenterCrop(crop),
            transforms.Normalize(mean, std),
        ],
    )


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    mode: str,
    num_workers: int,
) -> DataLoader:
    shuffle, drop_last = True, True
    if mode == 'val':
        shuffle, drop_last = False, False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
        num_workers=num_workers,
    )
