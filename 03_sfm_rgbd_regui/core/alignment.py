# -*- coding: utf-8 -*-
"""
Alignment algorithms for SFM-RGBD point cloud registration.

Provides rough alignment, iterative NUV rotation, and ICP refinement
with callback support for real-time UI updates.
"""

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import open3d as o3d
from scipy.signal import find_peaks

import utils.linear_algebra as util_la
import utils.cross_align as util_ca
import utils.pin_center as util_pc
import utils.icp_align as util_ia


@dataclass
class AlignmentParams:
    """
    Parameters for alignment algorithms.

    Parameters
    ----------
    search_radius : float
        Pin neighbor search radius in meters.
    cross_buffer : float
        Cross-strip buffer thickness in meters.
    icp_threshold : float
        ICP distance threshold per step in meters.
    icp_iter_num : int
        Number of ICP iterations.
    geometry_weight : float
        Weight for geometry vs color in ICP (0-1).

    Examples
    --------
    >>> params = AlignmentParams()
    >>> params.search_radius
    0.03
    """

    search_radius: float = 0.03
    cross_buffer: float = 0.001
    icp_threshold: float = 0.001
    icp_iter_num: int = 0
    geometry_weight: float = 0.1


@dataclass
class AlignmentResult:
    """
    Result from alignment computation.

    Attributes
    ----------
    transform_matrix : np.ndarray
        4x4 transformation matrix.
    rmse : float
        Root mean square error after alignment.
    open3d_rmse : float
        Open3D inlier RMSE from ICP.
    peak_angles : np.ndarray
        Detected local minimum angles in NUV rotation.
    rmse_curve : tuple
        (angles, rmses) arrays for RMSE curve plotting.
    selected_peak_idx : int
        Currently selected peak index.
    sfm_pin_data : dict
        SfM pin detection results.
    rgbd_pin_data : dict
        RGBD pin detection results.
    """

    transform_matrix: np.ndarray
    rmse: float = 0.0
    open3d_rmse: float = 0.0
    peak_angles: np.ndarray = field(default_factory=lambda: np.array([]))
    rmse_curve: tuple = field(default_factory=lambda: (np.array([]), np.array([])))
    selected_peak_idx: int = 0
    sfm_pin_data: dict = field(default_factory=dict)
    rgbd_pin_data: dict = field(default_factory=dict)
    nuv_matrices: list = field(default_factory=list)


class Aligner:
    """
    Registration algorithm with callback support for real-time UI updates.

    Parameters
    ----------
    params : AlignmentParams
        Alignment parameters.
    on_update : Callable, optional
        Callback function called when alignment updates occur.
        Signature: on_update(stage: str, data: dict).

    Attributes
    ----------
    params : AlignmentParams
        Current alignment parameters.

    Examples
    --------
    >>> params = AlignmentParams(search_radius=0.03)
    >>> aligner = Aligner(params)
    >>> result = aligner.compute_full_alignment(rgbd_data, sfm_data)
    """

    def __init__(
        self,
        params: Optional[AlignmentParams] = None,
        on_update: Optional[Callable] = None,
    ) -> None:
        """Initialize aligner with parameters and optional callback."""
        self.params = params or AlignmentParams()
        self._on_update = on_update
        self._last_result: Optional[AlignmentResult] = None

    def _notify(self, stage: str, data: dict) -> None:
        """Call update callback if registered."""
        if self._on_update is not None:
            self._on_update(stage, data)

    def find_pin_centers(
        self,
        rgbd_data: dict,
        sfm_data: dict,
    ) -> tuple[dict, dict]:
        """
        Find pin center and normal vector for both point clouds.

        Parameters
        ----------
        rgbd_data : dict
            RGBD data from DataLoader.load_rgbd().
        sfm_data : dict
            SfM data from DataLoader.load_sfm().

        Returns
        -------
        tuple[dict, dict]
            (sfm_pin_data, rgbd_pin_data) with center and normal info.
        """
        self._notify("pin_detection", {"status": "starting"})

        sfm_pin_data = util_pc.find_pin_center(
            sfm_data["pin_pcd"],
            sfm_data["pcd"],
            circle_color=[0, 0, 0],
            visualize=True,
            show=False,
            label="sfm",
        )
        rgbd_pin_data = util_pc.find_pin_center(
            rgbd_data["pin_pcd"],
            rgbd_data["pcd"],
            circle_color=[0, 0, 0],
            visualize=True,
            show=False,
            label="rgbd",
        )

        self._notify("pin_detection", {"status": "complete"})
        return sfm_pin_data, rgbd_pin_data

    def compute_rough_alignment(
        self,
        rgbd_pin_data: dict,
        sfm_pin_data: dict,
    ) -> np.ndarray:
        """
        Compute rough alignment matrix using pin positions.

        Parameters
        ----------
        rgbd_pin_data : dict
            RGBD pin detection result.
        sfm_pin_data : dict
            SfM pin detection result.

        Returns
        -------
        np.ndarray
            4x4 initial transformation matrix.
        """
        self._notify("rough_alignment", {"status": "computing"})

        imatrix = util_la.create_rotational_transform_matrix(
            rgbd_pin_data["circle_center_3d"],
            rgbd_pin_data["vector"],
            sfm_pin_data["circle_center_3d"],
            sfm_pin_data["vector"],
        )

        self._notify("rough_alignment", {"status": "complete", "matrix": imatrix})
        return imatrix

    def compute_nuv_rotation(
        self,
        rgbd_data: dict,
        sfm_data: dict,
        sfm_pin_data: dict,
        initial_matrix: np.ndarray,
    ) -> dict:
        """
        Compute iterative NUV rotation optimization.

        Parameters
        ----------
        rgbd_data : dict
            RGBD data dictionary.
        sfm_data : dict
            SfM data dictionary.
        sfm_pin_data : dict
            SfM pin detection result.
        initial_matrix : np.ndarray
            Initial transformation matrix.

        Returns
        -------
        dict
            Dictionary with angles, rmses, matrices, peaks.
        """
        self._notify("nuv_rotation", {"status": "starting"})

        # Transform RGBD data by initial matrix
        rgbd_pcd_it = copy.deepcopy(rgbd_data["pcd"]).transform(initial_matrix)
        rgbd_data_it = {
            "pcd": rgbd_pcd_it,
            "pin_idx": rgbd_data["pin_idx"],
            "pin_pcd": rgbd_pcd_it.select_by_index(rgbd_data["pin_idx"]),
        }

        rgbd_pin_data_it = util_pc.find_pin_center(
            rgbd_data_it["pin_pcd"],
            rgbd_data_it["pcd"],
            circle_color=[0, 0, 0],
            visualize=False,
            show=False,
        )

        # Find pin neighbors
        sfm_nbr_data = util_ia.find_pin_nbr(
            sfm_data,
            sfm_pin_data,
            self.params.search_radius,
            visualize=False,
            label="sfm-pin",
        )
        rgbd_nbr_data_it = util_ia.find_pin_nbr(
            rgbd_data_it,
            rgbd_pin_data_it,
            radius=self.params.search_radius,
        )

        # Iterative NUV rotation
        angles, rmses, matrices = util_ca.iterative_nuv_rotate(
            source_pcd=rgbd_nbr_data_it["nbr_pcd"],
            target_pcd=sfm_nbr_data["nbr_pcd"],
            rotate_point=sfm_pin_data["circle_center_3d"],
            normal_vector=sfm_pin_data["vector"],
            buffer=self.params.cross_buffer,
        )

        # Find local minima (peaks in negative)
        rmses_dup = np.append(rmses, rmses[0:3])
        peaks, _ = find_peaks(-rmses_dup, distance=1)
        peaks = peaks % len(rmses)
        peaks = np.unique(peaks)

        # Sort peaks by RMSE values
        if len(peaks) > 1:
            peak_values = rmses[peaks]
            order = np.argsort(peak_values)
            peaks = peaks[order]

        result = {
            "angles": angles,
            "rmses": rmses,
            "matrices": matrices,
            "peaks": peaks,
            "initial_matrix": initial_matrix,
        }

        self._notify("nuv_rotation", {"status": "complete", "peaks": len(peaks)})
        return result

    def compute_icp_refinement(
        self,
        rgbd_data: dict,
        sfm_data: dict,
        input_matrix: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Compute class-based ICP refinement.

        Parameters
        ----------
        rgbd_data : dict
            RGBD data dictionary.
        sfm_data : dict
            SfM data dictionary.
        input_matrix : np.ndarray
            Input transformation matrix from NUV step.

        Returns
        -------
        tuple[np.ndarray, float]
            (refined_matrix, open3d_rmse).
        """
        self._notify("icp", {"status": "starting", "iter": self.params.icp_iter_num})

        # Create binary point clouds for ICP
        sfm_pcd_bin = util_ia.paint_pcd_binary(sfm_data["pcd"], sfm_data["pin_idx"])
        rgbd_pcd_bin = util_ia.paint_pcd_binary(rgbd_data["pcd"], rgbd_data["pin_idx"])

        tmatrix, o3d_rmse = util_ia.color_based_icp(
            rgbd_pcd_bin,
            sfm_pcd_bin,
            input_matrix,
            threshold=self.params.icp_threshold,
            max_iter=self.params.icp_iter_num,
            geometry_weight=self.params.geometry_weight,
            return_rmse=True,
        )

        self._notify("icp", {"status": "complete", "rmse": o3d_rmse})
        return tmatrix, o3d_rmse

    def compute_full_alignment(
        self,
        rgbd_data: dict,
        sfm_data: dict,
        selected_peak: int = 0,
    ) -> AlignmentResult:
        """
        Compute full alignment pipeline.

        Parameters
        ----------
        rgbd_data : dict
            RGBD data from DataLoader.load_rgbd().
        sfm_data : dict
            SfM data from DataLoader.load_sfm().
        selected_peak : int
            Index of selected peak in local minima.

        Returns
        -------
        AlignmentResult
            Complete alignment result with matrix and metadata.
        """
        # Step 1: Find pin centers
        sfm_pin_data, rgbd_pin_data = self.find_pin_centers(rgbd_data, sfm_data)

        # Step 2: Rough alignment
        imatrix = self.compute_rough_alignment(rgbd_pin_data, sfm_pin_data)

        # Step 3: NUV rotation optimization
        nuv_result = self.compute_nuv_rotation(
            rgbd_data,
            sfm_data,
            sfm_pin_data,
            imatrix,
        )

        # Select peak
        peaks = nuv_result["peaks"]
        peak_idx = min(selected_peak, len(peaks) - 1)
        best_idx = peaks[peak_idx]
        iimatrix = nuv_result["matrices"][best_idx] @ imatrix

        # Step 4: ICP refinement
        tmatrix, o3d_rmse = self.compute_icp_refinement(rgbd_data, sfm_data, iimatrix)

        # Compute final RMSE
        source_pcd = copy.deepcopy(rgbd_data["pcd"]).transform(tmatrix)
        rmse = util_la.compute_distance_rmse(source_pcd, sfm_data["pcd"])

        result = AlignmentResult(
            transform_matrix=tmatrix,
            rmse=rmse,
            open3d_rmse=o3d_rmse,
            peak_angles=nuv_result["angles"][peaks],
            rmse_curve=(nuv_result["angles"], nuv_result["rmses"]),
            selected_peak_idx=peak_idx,
            sfm_pin_data=sfm_pin_data,
            rgbd_pin_data=rgbd_pin_data,
            nuv_matrices=nuv_result["matrices"],
        )

        self._last_result = result
        return result

    def recompute_with_peak(
        self,
        peak_idx: int,
        rgbd_data: dict,
        sfm_data: dict,
    ) -> AlignmentResult:
        """
        Recompute ICP with a different peak selection.

        Uses cached NUV results for efficiency.

        Parameters
        ----------
        peak_idx : int
            New peak index to use.
        rgbd_data : dict
            RGBD data dictionary.
        sfm_data : dict
            SfM data dictionary.

        Returns
        -------
        AlignmentResult
            Updated alignment result.
        """
        if self._last_result is None:
            return self.compute_full_alignment(rgbd_data, sfm_data, peak_idx)

        # Reuse cached NUV data
        last = self._last_result
        angles, rmses = last.rmse_curve
        peaks_mask = np.isin(angles, last.peak_angles)
        peak_indices = np.where(peaks_mask)[0]

        if peak_idx >= len(peak_indices):
            peak_idx = len(peak_indices) - 1

        best_angle_idx = peak_indices[peak_idx]
        nuv_matrix = last.nuv_matrices[best_angle_idx]

        # Recompute rough alignment
        imatrix = self.compute_rough_alignment(
            last.rgbd_pin_data,
            last.sfm_pin_data,
        )
        iimatrix = nuv_matrix @ imatrix

        # Recompute ICP
        tmatrix, o3d_rmse = self.compute_icp_refinement(rgbd_data, sfm_data, iimatrix)

        source_pcd = copy.deepcopy(rgbd_data["pcd"]).transform(tmatrix)
        rmse = util_la.compute_distance_rmse(source_pcd, sfm_data["pcd"])

        return AlignmentResult(
            transform_matrix=tmatrix,
            rmse=rmse,
            open3d_rmse=o3d_rmse,
            peak_angles=last.peak_angles,
            rmse_curve=last.rmse_curve,
            selected_peak_idx=peak_idx,
            sfm_pin_data=last.sfm_pin_data,
            rgbd_pin_data=last.rgbd_pin_data,
            nuv_matrices=last.nuv_matrices,
        )

    def update_params(self, params: AlignmentParams) -> None:
        """
        Update alignment parameters.

        Parameters
        ----------
        params : AlignmentParams
            New parameters to use.
        """
        self.params = params
