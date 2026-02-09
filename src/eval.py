"""
Script for evaluating single map. Main function is imported for other tasks.
"""
import tempfile
import os
import glob
import pandas as pd
import torch
import shutil
from omegaconf import DictConfig
from src.evaluation_utils import prepare_input, merge_output
import hydra
from src.utils import RankedLogger
import json


logger = RankedLogger(__name__)


def _prepare_data(config: DictConfig) -> None:
    # Read the paths
    input_map_file_path = os.path.abspath(config.input_map_path)
    output_map_file_path = os.path.abspath(config.output_map_path)

    # Sanity check
    if not os.path.exists(input_map_file_path):
        logger.info(f"Error: Given map path does not exist: {input_map_file_path}")
        return None
    if os.path.exists(output_map_file_path):
        logger.info(f"Skip. Output map already exist: {output_map_file_path}")
        return None
    
    # Setup tempdir
    temp_dir = tempfile.mkdtemp()
    temp_input_dir = os.path.join(temp_dir, "input")
    temp_output_dir = os.path.join(temp_dir, "output")
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)

    return temp_dir, temp_input_dir, temp_output_dir, input_map_file_path, output_map_file_path


def _prepare_predict_df(
    predict_dataset_dir: str,
    input_map_file_path: str
):
    filelist = glob.glob(os.path.join(predict_dataset_dir, '*.mrc'))

    return pd.DataFrame({
        'input_path': filelist, 
        'input_map_path': [input_map_file_path] * len(filelist),
    })

    
def eval(config: DictConfig) -> None:
    # I. Set-up the data
    print(config.input_map_path)
    logger.info('Preparing data...')
    res = _prepare_data(config)
    if res is None:
        return
    temp_dir, temp_input_dir, temp_output_dir, input_map_file_path, output_map_file_path = res

    output_dir = os.path.dirname(output_map_file_path)
    os.makedirs(output_dir, exist_ok=True)

    try:

        # II. Set-up the model
        logger.info(f'Instantiating model from <{config.model._target_}>')
        model = hydra.utils.instantiate(config.model, _recursive_=False)

        # III. Run this pseudo-loop
        # a) Prepare input
        logger.info(f'Preparing input from {input_map_file_path} to tmpdir: {temp_input_dir}')
        prepare_input(input_map_file_path, temp_input_dir)

        # b) Prepare dataframe 
        predict_df = _prepare_predict_df(temp_input_dir, input_map_file_path)
        logger.info(f'Created {len(predict_df)} files for enhancement.')

        # b) Set-up the datamodule
        logger.info(f'Instantiating datamodule from <{config.data._target_}>')
        datamodule = hydra.utils.instantiate(config.data, predict_df=predict_df)

        # c) Set-up the prediction writer
        logger.info(f'Instantiating prediction writer from <{config.prediction_writer._target_}>')
        config.prediction_writer['output_dir'] = temp_output_dir
        prediction_writer = hydra.utils.instantiate(config.prediction_writer)

        # d) Set-up the trainer
        logger.info(f'Instantiating trainer from <{config.trainer._target_}>')
        config.trainer['logger'] = False
        trainer = hydra.utils.instantiate(config.trainer, callbacks=[prediction_writer])

        # e) Run the prediction phase
        trainer.predict(model, datamodule=datamodule, ckpt_path=config.ckpt_path)

        # f) Merge output
        logger.info(f'Merging outputs, tmp input: {temp_input_dir} -> tmp output: {temp_output_dir}, map input: {input_map_file_path} -> map output: {output_map_file_path}')
        success = merge_output(temp_output_dir, output_map_file_path, temp_input_dir, input_map_file_path)

    except Exception as e:
        import traceback
        logger.error(str(traceback.format_exc()))
    finally:
        logger.info(f'Deleting temp dir files: {temp_dir}')
        shutil.rmtree(temp_dir, ignore_errors=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_singlemap.yaml")
def main(config: DictConfig) -> None:
    eval(config)
    

if __name__ == '__main__':
    main()
