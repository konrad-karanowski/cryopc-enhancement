"""
This script is for evaluation on the whole dataset.
"""
import os
import logging

import tqdm
import hydra
import pandas as pd
from omegaconf import DictConfig

from src.eval import eval
from src.utils import RankedLogger


logger = RankedLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_testset.yaml")
def main(config: DictConfig) -> None:
    """Main function :3

    Args:
        config (DictConfig): The same config as for eval_singlemap, but it has additional fields: csv path, collection_dir and output_dir (where to save files).
    """
    # I. Read csv 
    df = pd.read_csv(config.csv_path)
    emdb_names = df.emdb_entry.unique()

    # II. Set-up the collection dir
    collection_dir = os.path.abspath(config.collection_dir)
    output_dir = os.path.abspath(config.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    for i, map in (pbar := tqdm.tqdm(enumerate(emdb_names), total=len(emdb_names))):
        emdb_id = str(map)
        
        map_path = os.path.join(collection_dir, "emd_" + emdb_id, "cryoem_deposited.mrc")
        if not os.path.exists(map_path):
            pbar.set_description("Skipping EMD {embd_id} {map_path} does not exist.")
            continue

        pbar.set_description(f"Processing EMD {emdb_id}")
        emdb_output_dir = os.path.join(output_dir, "emd_" + emdb_id)
        os.makedirs(emdb_output_dir, exist_ok=True)
        output_map_path = os.path.join(emdb_output_dir, f"{config.map_save_prefix}_{emdb_id}.mrc")

        # cheat to update paths
        config['input_map_path'] = map_path
        config['output_map_path'] = output_map_path

        eval(config)
        pbar.set_description("-------------------------------------------------------------------------------------------------")


if __name__ == '__main__':
    main()
