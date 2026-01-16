# -*- coding: utf-8 -*-
"""
IO utilities for JSON result files.

Handles saving and loading alignment results in the specified format.
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder with numpy type support.

    Converts numpy types to native Python types for JSON serialization.
    """

    def default(self, obj: Any) -> Any:
        """Handle numpy types in JSON encoding."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_result_json(
    output_path: Union[str, Path],
    rgbd_pcd_file: str,
    sfm_mesh_file: str,
    transform_matrix: np.ndarray,
    rmse: float,
    open3d_rmse: float,
    sfm_pin_data: dict,
    rgbd_pin_data: dict,
    search_radius: float,
    cross_buffer: float,
    icp_iter_num: int,
    icp_threshold: float,
    geometry_weight: float,
    hsv_weight: Optional[list] = None,
    hsv_denoise_threshold: Optional[float] = None,
    hsv_denoised_volume: Optional[float] = None,
    peak_angles: Optional[Union[list, np.ndarray]] = None,
    selected_peak_idx: Optional[int] = None,
) -> None:
    """
    Save alignment result to JSON file.

    Parameters
    ----------
    output_path : Path
        Output JSON file path.
    rgbd_pcd_file : str
        Relative path to RGBD point cloud file.
    sfm_mesh_file : str
        Relative path to SfM mesh file.
    transform_matrix : np.ndarray
        4x4 transformation matrix.
    rmse : float
        Root mean square minimum distance.
    open3d_rmse : float
        Open3D inlier RMSE.
    sfm_pin_data : dict
        SfM pin detection data.
    rgbd_pin_data : dict
        RGBD pin detection data.
    search_radius : float
        Pin neighbor search radius.
    cross_buffer : float
        Cross-strip buffer thickness.
    icp_iter_num : int
        Number of ICP iterations.
    icp_threshold : float
        ICP distance threshold.
    geometry_weight : float
        ICP geometry weight.
    hsv_weight : list, optional
        HSV color weights for SfM.
    hsv_denoise_threshold : float, optional
        HSV index denoise threshold.
    hsv_denoised_volume : float, optional
        HSV denoised volume.
    peak_angles : list or np.ndarray, optional
        List of potential local minima angles.
    selected_peak_idx : int, optional
        Index of the selected peak in the angles list.

    Examples
    --------
    >>> save_result_json(
    ...     Path("/output/2R1-1.json"),
    ...     "1_rgbd/2_pcd/2R1-1/2R1-1_pcd_362.ply",
    ...     "2_sfm/1_mesh/2R1-1/2R1-1.obj",
    ...     transform_matrix,
    ...     rmse=0.001,
    ...     open3d_rmse=0.0004,
    ...     sfm_pin_data=sfm_pin,
    ...     rgbd_pin_data=rgbd_pin,
    ...     search_radius=0.03,
    ...     cross_buffer=0.001,
    ...     icp_iter_num=10,
    ...     icp_threshold=0.0005,
    ...     geometry_weight=0.1,
    ...     peak_angles=[10, 100, 200],
    ...     selected_peak_idx=1,
    ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "rgbd_pcd_file": rgbd_pcd_file,
        "sfm_mesh_file": sfm_mesh_file,
        "T": transform_matrix,
        "rms_minimum_distance": rmse,
        "open3d_inlier_rmse": open3d_rmse,
        "meta": {
            "pin_segment": {
                "sfm": {
                    "hsv_weight": hsv_weight or [0.5, 0.1, 0.3],
                    "hsv_index_denoise_threshold": hsv_denoise_threshold or 0.2,
                    "hsv_index_denoised_volume": hsv_denoised_volume or 0.0,
                    "center": _get_center(sfm_pin_data),
                    "radius(m)": sfm_pin_data.get("circle_radius", 0.0),
                    "normal_vector": _get_vector(sfm_pin_data),
                },
                "rgbd": {
                    "center": _get_center(rgbd_pin_data),
                    "radius(m)": rgbd_pin_data.get("circle_radius", 0.0),
                    "normal_vector": _get_vector(rgbd_pin_data),
                },
            },
            "pin_neighbor": {
                "search_radius(m)": search_radius,
                "corss_buffer(m)": cross_buffer,
            },
            "class_based_icp": {
                "iter_num": icp_iter_num,
                "iter_distance(m)": icp_threshold,
                "geometry_weight": geometry_weight,
            },
            "rms_analysis": {
                "potential_local_minima": peak_angles if peak_angles is not None else [],
                "selected": selected_peak_idx if selected_peak_idx is not None else 0,
            },
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, cls=NumpyEncoder, indent=4)


def _get_center(pin_data: dict) -> list:
    """Extract center from pin data."""
    center = pin_data.get("circle_center_3d", [0, 0, 0])
    if isinstance(center, np.ndarray):
        return center.tolist()
    return list(center)


def _get_vector(pin_data: dict) -> list:
    """Extract normal vector from pin data."""
    vector = pin_data.get("vector", [0, 0, 1])
    if isinstance(vector, np.ndarray):
        return vector.tolist()
    return list(vector)


def load_result_json(json_path: Union[str, Path]) -> dict:
    """
    Load alignment result from JSON file.

    Parameters
    ----------
    json_path : Path
        Path to JSON file.

    Returns
    -------
    dict
        Loaded result with transform matrix as numpy array.

    Examples
    --------
    >>> result = load_result_json(Path("/output/2R1-1.json"))
    >>> result["T"].shape
    (4, 4)
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert transform matrix to numpy array
    if "T" in data:
        data["T"] = np.array(data["T"])

    return data
