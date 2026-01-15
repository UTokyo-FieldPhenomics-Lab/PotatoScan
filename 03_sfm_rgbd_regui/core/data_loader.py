# -*- coding: utf-8 -*-
"""
Data loader module for RGBD and SfM point cloud data.

Provides unified data loading with configurable paths, refactored from
the original RgbdPinFetcher and SfMPinFetcher classes.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from utils.pin_segment import RgbdPinFetcher, SfMPinFetcher


@dataclass
class DataConfig:
    """
    Configuration for dataset paths.

    Parameters
    ----------
    dataset_root : Path
        Root folder for the dataset (contains 1_rgbd, 2_sfm, 3_pair).
    output_folder : Path, optional
        Custom output folder for transform matrices.
        Defaults to dataset_root / "3_pair/tmatrix".

    Notes
    -----
    Pin reference folder is fixed at dataset_root / 2_sfm / 3_pin_refs.
    CSV file is at dataset_root / 3_pair / ground_truth_2025.csv.

    Examples
    --------
    >>> config = DataConfig(dataset_root=Path("/data/3DPotatoTwin"))
    >>> config.output_folder
    PosixPath('/data/3DPotatoTwin/3_pair/tmatrix')
    """

    dataset_root: Path
    output_folder: Optional[Path] = None

    def __post_init__(self) -> None:
        """Set default output folder if not specified."""
        if self.output_folder is None:
            self.output_folder = self.dataset_root / "3_pair/tmatrix"

    @property
    def rgbd_folder(self) -> Path:
        """Path to RGBD image folder."""
        return self.dataset_root / "1_rgbd/1_image"

    @property
    def sfm_folder(self) -> Path:
        """Path to SfM point cloud folder."""
        return self.dataset_root / "2_sfm/2_pcd"

    @property
    def csv_file(self) -> Path:
        """Path to ground truth CSV file."""
        return self.dataset_root / "3_pair/ground_truth_2025.csv"


class DataLoader:
    """
    Unified data loader for RGBD and SfM point clouds.

    Wraps the original RgbdPinFetcher and SfMPinFetcher with a cleaner
    interface and configurable paths.

    Parameters
    ----------
    config : DataConfig
        Configuration object with dataset paths.

    Attributes
    ----------
    config : DataConfig
        Stored configuration object.

    Examples
    --------
    >>> config = DataConfig(dataset_root=Path("/data"))
    >>> loader = DataLoader(config)
    >>> ids = loader.get_ids()
    >>> rgbd_data = loader.load_rgbd("2R1-1")
    """

    def __init__(self, config: DataConfig) -> None:
        """Initialize data loader with configuration."""
        self.config = config
        self._rgbd_fetcher: Optional[RgbdPinFetcher] = None
        self._sfm_fetcher: Optional[SfMPinFetcher] = None

    def _init_fetchers(self) -> None:
        """Lazily initialize fetcher objects."""
        if self._rgbd_fetcher is None:
            self._rgbd_fetcher = RgbdPinFetcher(self.config.dataset_root)
        if self._sfm_fetcher is None:
            self._sfm_fetcher = SfMPinFetcher(
                self.config.dataset_root,
                self.config.csv_file,
            )

    def get_ids(self) -> list[str]:
        """
        Get list of all potato IDs in the dataset.

        Returns
        -------
        list[str]
            Sorted list of potato IDs (folder names in SfM folder).

        Examples
        --------
        >>> loader.get_ids()
        ['2R1-1', '2R1-2', '2R1-3', ...]
        """
        ids = []
        for folder in self.config.sfm_folder.glob("*"):
            if folder.is_dir():
                ids.append(folder.name)
        return sorted(ids)

    def get_completed_ids(self) -> list[str]:
        """
        Get list of IDs that have completed registration.

        Returns
        -------
        list[str]
            List of IDs with existing JSON matrix files.
        """
        completed = []
        if not self.config.output_folder.exists():
            return completed

        for json_file in self.config.output_folder.glob("*.json"):
            completed.append(json_file.stem)
        return completed

    def load_rgbd(
        self,
        pid: str,
        img_id: Optional[str] = None,
        visualize: bool = False,
    ) -> dict:
        """
        Load RGBD point cloud data for a potato ID.

        Parameters
        ----------
        pid : str
            Potato ID to load.
        img_id : str, optional
            Specific RGBD image ID. If None, uses center image.
        visualize : bool, default False
            Whether to prepare visualization geometries.

        Returns
        -------
        dict
            Dictionary containing:
            - 'pcd': o3d.geometry.PointCloud
            - 'pin_pcd': o3d.geometry.PointCloud
            - 'pin_idx': np.ndarray
            - 'pcd_rela_path': str
            - Additional visualization data if visualize=True.
        """
        self._init_fetchers()
        return self._rgbd_fetcher.get(
            pid,
            img_id=img_id,
            visualize=visualize,
            show=False,
        )

    def load_sfm(
        self,
        pid: str,
        visualize: bool = False,
    ) -> dict:
        """
        Load SfM point cloud data for a potato ID.

        Parameters
        ----------
        pid : str
            Potato ID to load.
        visualize : bool, default False
            Whether to prepare visualization geometries.

        Returns
        -------
        dict
            Dictionary containing:
            - 'pcd': o3d.geometry.PointCloud
            - 'pin_pcd': o3d.geometry.PointCloud
            - 'pin_idx': np.ndarray
            - 'pcd_rela_path': str
            - 'hsv_weight': list
            - 'stop_thresh': float
            - Additional visualization data if visualize=True.
        """
        self._init_fetchers()
        return self._sfm_fetcher.get(
            pid,
            visualize=visualize,
            show=False,
        )

    def get_output_path(self, pid: str) -> Path:
        """
        Get the output JSON file path for a potato ID.

        Parameters
        ----------
        pid : str
            Potato ID.

        Returns
        -------
        Path
            Path to the output JSON file.
        """
        return self.config.output_folder / f"{pid}.json"

    def is_completed(self, pid: str) -> bool:
        """
        Check if a potato ID has completed registration.

        Parameters
        ----------
        pid : str
            Potato ID to check.

        Returns
        -------
        bool
            True if a JSON file exists for this ID.
        """
        return self.get_output_path(pid).exists()
