from typing import Optional

import tempfile
import mrcfile
import empatches
import os
# import scripts.utils.map_processing_utils as cu
import numpy as np
from copy import deepcopy
import torch
import glob
from iotbx.data_manager import DataManager
from cctbx import uctbx
import iotbx.mrcfile as iotbxmrcfile
from src.utils import RankedLogger

emp = empatches.EMPatches()

logger = RankedLogger(__name__)


def prepare_input(
        input_map_file_path: str, 
        temp_input_dir: str
    ) -> None:
    with mrcfile.open(input_map_file_path) as mrc:
        mapdata = mrc.data.astype(np.float32).copy()
    
    # Normalize the input
    percentile_99p999 = np.percentile(mapdata[np.nonzero(mapdata)], 99.999)
    mapdata /= percentile_99p999
    mapdata[mapdata < 0] = 0
    mapdata[mapdata > 1] = 1

    blocks, indices = emp.extract_patches(mapdata, patchsize=48, stride=38, vox=True)

    for i in range(0, len(blocks)):
        input_block_filepath = os.path.join(temp_input_dir, str(i)+".mrc")

        with mrcfile.new(input_block_filepath, overwrite=True) as mrc:
            mrc.set_data(deepcopy(blocks[i]))

    block_indices_file_path = os.path.join(temp_input_dir, "block_indices.pt")
    torch.save(indices, block_indices_file_path, pickle_protocol=5)

def merge_output(
        temp_output_dir: str, 
        output_map_file_path: str, 
        temp_input_dir: str, 
        input_map_file_path: str
    ) -> bool:
    prediction_output_files = glob.glob(os.path.join(temp_output_dir, "predictions_output_*.pt"))
    predictions_filename_files = glob.glob(os.path.join(temp_output_dir, "predictions_filename_*.pt"))
    block_indices_ref = torch.load(os.path.join(temp_input_dir, "block_indices.pt"))

    if len(prediction_output_files) != len(predictions_filename_files):
        logger.error("Output files and filename files don't match")
        return False

    preds = [None]*len(block_indices_ref)
    indices = [None]*len(block_indices_ref)

    for i in range(len(prediction_output_files)):
        predictions_partial = torch.load(os.path.join(temp_output_dir, "predictions_output_"+str(i)+".pt"))
        filename_partial = torch.load(os.path.join(temp_output_dir, "predictions_filename_"+str(i)+".pt"))

        if len(predictions_partial) != len(filename_partial):
            logger.error("number of predictions and indices doesnt match")
            return False

        for i in range(len(predictions_partial)):
            cube_index = int(filename_partial[i].split("/")[-1].split(".")[0])
            preds[cube_index] = predictions_partial[i]
            indices[cube_index] = block_indices_ref[cube_index]

    merged_arr = emp.merge_patches(preds, indices, mode="avg")

    # WORKS FOR ALL MAPS EXCEPT EMD-22845. BELOW APPROACH FIXES IT
    # ------------------------------------------------------------
    # shutil.copyfile(input_map_file_path, output_map_file_path)
    # with mrcfile.open(output_map_file_path, mode='r+') as mrc:
    #     mrc.set_data(merged_arr)
    
    # NEW APPROACH. YET TO CONFIRM IT WORKS FOR ALL MAPS. BUT FIXES EMD-22845
    # -----------------------------------------------------------------------
    with mrcfile.open(input_map_file_path, mode='r+') as mrc:
        voxel_size = np.array(mrc.voxel_size.data)
        axis_order = [mrc.header.mapc, mrc.header.mapr, mrc.header.maps]
        origin = np.array(mrc.nstart.data)
        unit_cell = np.array(mrc.header.cella)
        
    # unit_cell = [merged_arr.shape[0] * voxel_size['x'], merged_arr.shape[1] * voxel_size['y'], merged_arr.shape[2] * voxel_size['z']]
    tmpfile = tempfile.NamedTemporaryFile(suffix='.mrc').name
    with mrcfile.new(tmpfile, overwrite=True) as mrc:
        mrc.set_data(merged_arr)
        mrc.voxel_size = voxel_size
        mrc.header.mapc = axis_order[0]
        mrc.header.mapr = axis_order[1]
        mrc.header.maps = axis_order[2]
    
    dm = DataManager()
    map_inp = dm.get_real_map(tmpfile)
    map_inp.shift_origin(desired_origin = [int(origin['x']), int(origin['y']), int(origin['z'])])
    iotbxmrcfile.write_ccp4_map(file_name=output_map_file_path,
        unit_cell=uctbx.unit_cell([float(unit_cell['x']), float(unit_cell['y']), float(unit_cell['z'])]),
        map_data=map_inp.map_data(),
        output_axis_order=axis_order
    )
    
    return True
