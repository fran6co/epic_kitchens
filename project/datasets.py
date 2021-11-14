import csv
import json
import os
import random
from itertools import chain as chain

import numpy as np
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager
from timesformer.datasets import utils as utils


class Epickitchens(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        mode,
        mean,
        std,
        num_frames,
        num_clips,
        jitter_scales,
        crop_size,
        num_spatial_crops,
    ):
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Epickitchens".format(mode)
        self.mode = mode

        self._video_meta = {}
        self._num_clips = num_clips
        self._jitter_scales = jitter_scales
        self._crop_size = crop_size
        self._num_spatial_crops = num_spatial_crops
        self._num_frames = num_frames
        self._mean = mean
        self._std = std
        self._random_flip = True
        self._data_path = data_path

        self._construct_loader()

    def _construct_loader(self):
        # Loading labels.
        label_file = os.path.join(
            self._data_path,
            "{}_action_classes.csv".format("train" if self.mode == "train" else "val"),
        )
        with PathManager.open(label_file, "r") as f:
            labels = [r for r in csv.DictReader(f)]

        self._video_names = []
        self._labels = []
        self._path_to_videos = []
        label_dict = json.load(
            open(os.path.join(self._data_path, "class_names.json"), "r")
        )
        for video in labels:
            label = int(label_dict[video["action_class"].strip()])
            self._video_names.append(video["video_id"].strip())
            self._labels.append(label)

            video_path = os.path.join(
                self._data_path,
                "train" if self.mode == "train" else "val",
                video["participant_id"].strip(),
                video["video_id"].strip(),
            )

            self._path_to_videos.append(
                [os.path.join(video_path, p) for p in os.listdir(video_path)]
            )

        # Extend self when self._num_clips > 1 (during testing).
        self._path_to_videos = list(
            chain.from_iterable([[x] * self._num_clips for x in self._path_to_videos])
        )
        self._labels = list(
            chain.from_iterable([[x] * self._num_clips for x in self._labels])
        )
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [range(self._num_clips) for _ in range(len(self._path_to_videos))]
            )
        )

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, _ = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self._jitter_scales[0]
            max_scale = self._jitter_scales[1]
            crop_size = self._crop_size
        elif self.mode in ["test"]:
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index] % self._num_spatial_crops
            )
            if self._num_spatial_crops == 1:
                spatial_sample_index = 1

            min_scale, max_scale, crop_size = [self._crop_size] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        label = self._labels[index]

        num_frames = self._num_frames
        video_length = len(self._path_to_videos[index])

        seg_size = float(video_length - 1) / num_frames
        seq = []
        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.mode == "train":
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)

        frames = torch.as_tensor(
            utils.retry_load_images(
                [self._path_to_videos[index][frame] for frame in seq]
            )
        )

        # Perform color normalization.
        frames = utils.tensor_normalize(frames, list(self._mean), list(self._std))

        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self._random_flip,
            inverse_uniform_sampling=False,
        )
        # Perform temporal sampling from the fast pathway.
        frames = torch.index_select(
            frames,
            1,
            torch.linspace(0, frames.shape[1] - 1, self._num_frames).long(),
        )
        return frames, label, index, {}

    def __len__(self):
        return len(self._path_to_videos)
