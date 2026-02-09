from typing import Tuple, Optional

import gzip
import os
import mrcfile
import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from monai.transforms import Randomizable, apply_transform
from monai.utils import MAX_SEED, get_seed
from monai.data.meta_tensor import MetaTensor
import glob

from src.data.data_utils import get_point_cloud, get_cube_coordinates


class PCCryoemDensityMapBlockPredictDataset(Dataset):

    def __init__(
        self, 
        df: pd.DataFrame,
        num_points: int,
        input_dir: str,
    ) -> None:
        self.df = df
        self.num_points = num_points
        self.block_indices = torch.load(os.path.join(input_dir, "block_indices.pt"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]

        X = torch.from_numpy(np.copy(mrcfile.mmap(sample.input_path).data))

        points = get_point_cloud(sample.input_point_clouds_path, self.num_points)

        # Images generally have 3 channels (R, G, B). But, in cryoem maps, its just 1 channel 3D image of shape (h, w, d)
        # so reshape it as (c=1, h, w, d)
        X = X.unsqueeze(0)

        output = {
            'X': X,
            'c': points,
            'file': sample.input_path,
        }
        return output


class PCCryoemDensityMapBlockDataset(Dataset):

    def __init__(
        self, 
        df: pd.DataFrame,
        num_points: int,
    ) -> None:
        self.df = df
        self.num_points = num_points

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]

        emdb_name = sample.emdb_name

        X = torch.from_numpy(np.copy(mrcfile.mmap(sample.input_path).data))
        y = torch.from_numpy(np.copy(mrcfile.mmap(sample.target_path).data))

        points = get_point_cloud(sample.input_point_clouds_path, self.num_points)

        # Images generally have 3 channels (R, G, B). But, in cryoem maps, its just 1 channel 3D image of shape (h, w, d)
        # so reshape it as (c=1, h, w, d)
        X = X.unsqueeze(0)
        y = y.unsqueeze(0)

        output = {
            'X': X,
            'y': y,
            'c': points,
            'filename': emdb_name,
        }
        return output


class PCCryoemDensityMapBlockAugmentedDataset(Dataset, Randomizable):
    def __init__(
        self,
        df: pd.DataFrame,
        num_points: int,
        i_transform,
        t_transform,
    ) -> None:
        self.df = df

        self.set_random_state(seed=get_seed())
        self._seed = 0  # transform synchronization seed
        self.i_transform = i_transform
        self.t_transform = t_transform

        # for normalization
        self.num_points = num_points

    def __len__(self):
        return len(self.df)

    def randomize(self):
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, idx):
        self.randomize()

        sample = self.df.iloc[idx]

        emdb_name = sample.emdb_name

        X = torch.from_numpy(np.copy(mrcfile.mmap(sample.input_path).data))
        y = torch.from_numpy(np.copy(mrcfile.mmap(sample.target_path).data))

        points = get_point_cloud(sample.input_point_clouds_path, self.num_points)

        # Images generally have 3 channels (R, G, B). But, in cryoem maps, its just 1 channel 3D image of shape (h, w, d)
        # so reshape it as (c=1, h, w, d)
        X = X.unsqueeze(0)
        y = y.unsqueeze(0)

        if self.i_transform is not None:
            if isinstance(self.i_transform, Randomizable):
                self.i_transform.set_random_state(seed=self._seed)
            X = apply_transform(self.i_transform, X, map_items=False)

        if self.t_transform is not None:
            if isinstance(self.t_transform, Randomizable):
                self.t_transform.set_random_state(seed=self._seed)
            y = apply_transform(self.t_transform, y, map_items=False)

        output = {
            'X': X,
            'y': y,
            'c': points,
            'filename': emdb_name,
        }

        return output
