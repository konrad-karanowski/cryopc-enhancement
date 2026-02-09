from typing import Tuple, Optional

import torch
import numpy as np
import mrcfile
import ast


def get_point_cloud(path: str, num_points: int) -> torch.Tensor:
    """
    Loads point cloud and samples random points.

    Parameters:
        path (str): Path to point cloud file.
        num_points (int): Number of points to sample.

    Returns:
        Tensor of size [features (usually 4), num_points].
    """
    p = torch.load(path)
    p = p[:, torch.randperm(p.size(1))[:num_points]]
    return p


def sample_point_clouds(
        map_path: str, 
        num_points: int, 
        threshold_k: float, 
        normalize_point_cloud_coords: bool = False
    ) -> torch.Tensor:
    """
    Loads an MRC map, thresholds it using mean + k * std, samples `num_points` from above-threshold voxels,
    and returns them as a tensor of shape [4, num_points]: [x, y, z, value].
    
    Parameters:
        map_path (str): Path to the .mrc file.
        num_points (int): Number of points to sample.
        threshold_k (float): Multiplier for standard deviation to compute threshold.
        normalize_point_cloud_coords (bool): If True, normalize (x, y, z) to [-1, 1] range.

    Returns:
        Tensor of shape [4, N] -> [(x, y, z, value), num_points].
    """
    # Step 1: Load map
    with mrcfile.open(map_path, permissive=True) as mrc:
        data = mrc.data.copy()


    # Step 2: Threshold
    threshold = data.mean() + threshold_k * data.std()
    mask = data > threshold
    coords = np.argwhere(mask)  # shape: [N, 3]

    if len(coords) == 0:
        raise ValueError("No voxels found above the threshold. Try lowering threshold_k.")

    # Step 3: Random sampling
    if len(coords) > num_points:
        indices = np.random.choice(len(coords), size=num_points, replace=False)
    else:
        indices = np.random.choice(len(coords), size=num_points, replace=True)

    sampled_coords = coords[indices]  # shape: [num_points, 3]

    # Step 4: Normalize coordinates if requested
    if normalize_point_cloud_coords:
        dims = np.array(data.shape)  # [Z, Y, X]
        center = (dims - 1) / 2.0
        scale = dims.max() / 2.0  # uniform scaling
        sampled_coords = (sampled_coords - center) / scale  # → [-1, 1] roughly

    # Step 5: Get values at those coordinates (must round + clip if normalized)
    if normalize_point_cloud_coords:
        # Convert back to int voxel indices for lookup
        voxel_coords = np.clip(np.round((sampled_coords * scale) + center).astype(int), 0, np.array(data.shape) - 1)
    else:
        voxel_coords = sampled_coords

    values = data[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]

    # Step 6: Stack and return as tensor [4, num_points]
    point_cloud = np.hstack([sampled_coords, values[:, None]])  # [N, 4]
    return torch.tensor(point_cloud.T, dtype=torch.float32)   # [4, N]


def get_cube_coordinates(coords: Tuple, sizes: Optional[Tuple] = None) -> torch.Tensor:
    """
    Convert cube coordinates to a tensor, optionally normalized by sizes.
    
    Args:
        coords: Tuple of 6 ints (x1, x2, y1, y2, z1, z2)
        sizes: Optional tuple of 3 ints (X_max, Y_max, Z_max) for normalization
        
    Returns:
        Tensor of shape [6] with coordinates, normalized if sizes provided
    """
    # Convert coords tuple to tensor
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    
    # If no sizes provided, return as-is
    if sizes is None:
        return coords_tensor
    
    # Normalize by sizes: (x1, x2, y1, y2, z1, z2) / (X_max, X_max, Y_max, Y_max, Z_max, Z_max)
    X_max, Y_max, Z_max = sizes
    normalization = torch.tensor([X_max, X_max, Y_max, Y_max, Z_max, Z_max], dtype=torch.float32)
    
    return coords_tensor / normalization
