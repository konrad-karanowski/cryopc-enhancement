from typing import Dict, Optional

import glob
import os
import ast

import torch
import pandas as pd
import tqdm
import mrcfile
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from src.data.cryoem_pc_map_datasets import (
    PCCryoemDensityMapBlockAugmentedDataset, # Train
    PCCryoemDensityMapBlockDataset, # Validation
    PCCryoemDensityMapBlockPredictDataset, # Prediction
)
from src.data.data_utils import sample_point_clouds

from monai.transforms import (
    Compose,
    RandRotate90,
    RandAxisFlip,
    RandSpatialCrop,
)


class PCCryoemDensityMapDataModule(LightningDataModule):


    COLUMNS_TO_RETYPE = ('map_size', 'coords')
    COLUMNS_WITH_PATHS = (
        'input_path',
        'target_path',
        'input_map_path',
        'target_map_path',
        'input_point_clouds_path',
        'target_point_clouds_path'
    )

    def __init__(
        self,
        data_root: str = "",
        csv_name: str = "",
        predict_df: Optional[pd.DataFrame] = None,
        batch_size=8,
        num_workers=1,
        train_max_samples=10000,
        val_max_samples=1000,
        pin_memory=False,
        persistent_workers: bool = True,  # keep workers between epochs
        prefetch_factor: int = 2,  # can help with buffering,
        num_points: int = 1024,
        threshold_k: float = 1.5,
        
        # CONDITIONINGS
        normalize_coords: bool = False, # whether to normalize coordinates, NOT POINT CLOUD COORDINATES!!!!!!!!!!!
        normalize_point_cloud_coords: bool = False, # USED ONLY FOR TESTING!!!!

        *args,
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_root = data_root
        self.predict_df = predict_df
        self.csv_name = csv_name
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_max_samples = train_max_samples
        self.val_max_samples = val_max_samples
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        
        self.num_points = num_points
        self.threshold_k = threshold_k
        self.normalize_coords = normalize_coords
        self.normalize_point_cloud_coords = normalize_point_cloud_coords

    # def prepare_data(self, *args, **kwargs):
    #     self._setup('')

    def _prepare_predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        basedir = os.path.dirname(df.iloc[0].input_path)

        # Find all unique maps
        unique_maps = pd.unique(df.input_map_path)
        map_ = {}
        map_dims_ = {}
        for map_path in tqdm.tqdm(unique_maps):
            points_savepath = os.path.join(basedir, f'{os.path.basename(map_path).split(".")[0]}.pt')
            if not os.path.exists(points_savepath):
                points = sample_point_clouds(
                    map_path=map_path,
                    num_points=self.num_points,
                    threshold_k=self.threshold_k,
                    normalize_point_cloud_coords=self.normalize_point_cloud_coords
                )
            torch.save(points, points_savepath)
            map_[map_path] = points_savepath

            # Handle dimensions
            map_dims_[map_path] = mrcfile.mmap(map_path).data.shape

        df['input_point_clouds_path'] = df.input_map_path.apply(lambda x: map_[x])
        df['map_size'] = df.input_map_path.apply(lambda x: map_dims_[x])

        # Handle coords
        indices = torch.load(os.path.join(basedir, 'block_indices.pt'))
        df['coords'] = indices # indices are list of tuples

        return df, basedir


    def setup(self, stage):
        if stage == "fit" or stage == "validate":

            df = pd.read_csv(os.path.join(self.data_root, 'dataset', self.csv_name))

            # eval literals
            for column in self.COLUMNS_TO_RETYPE:
                if column in df.columns:
                    df[column] = df[column].apply(ast.literal_eval)

            # extend the patchs
            for column in self.COLUMNS_WITH_PATHS:
                df[column] = df[column].apply(lambda x: os.path.join(self.data_root, x))
            
            df_train = df[df.split == 'train']
            df_val = df[df.split == 'val']

            df_train = df_train.sort_values(by='input_path')
            if self.train_max_samples > 0 and self.train_max_samples < len(df_train):
                df_train = df_train.iloc[:self.train_max_samples]
            
            df_val = df_val.sort_values(by='input_path')
            if self.val_max_samples > 0 and self.val_max_samples < len(df_val):
                df_val = df_val.iloc[:self.val_max_samples]

            i_transform = Compose(
                [
                    RandSpatialCrop(
                        [48, 48, 48],
                        max_roi_size=[48, 48, 48],
                        random_center=True,
                        random_size=False,
                    ),
                    RandRotate90(prob=0.4),
                    RandAxisFlip(prob=0.4),
                ]
            )
            t_transform = Compose(
                [
                    RandSpatialCrop(
                        [48, 48, 48],
                        max_roi_size=[48, 48, 48],
                        random_center=True,
                        random_size=False,
                    ),
                    RandRotate90(prob=0.4),
                    RandAxisFlip(prob=0.4),
                ]
            )

            self.train_set = PCCryoemDensityMapBlockAugmentedDataset(
                df=df_train,
                num_points=self.num_points,
                i_transform=i_transform,
                t_transform=t_transform,
            )
            self.val_set = PCCryoemDensityMapBlockDataset(
                df=df_val,
                num_points=self.num_points,
            )
            
        elif stage == "predict":
            assert self.predict_df is not None, f'No predict dataframe found.'

            new_predict_df, basedir = self._prepare_predict_df(self.predict_df)

            self.predict_set = PCCryoemDensityMapBlockPredictDataset(
                df=new_predict_df,
                num_points=self.num_points,
                input_dir=basedir,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor
        )
