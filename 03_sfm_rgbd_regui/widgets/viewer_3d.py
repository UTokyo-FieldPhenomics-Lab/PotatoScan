# -*- coding: utf-8 -*-
"""
PyVista-based 3D viewer widget for point cloud visualization.

Uses pyvistaqt for native Qt embedding with interactive controls.
"""

from typing import Optional

import numpy as np
import open3d as o3d
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class Viewer3D(QWidget):
    """
    3D point cloud viewer with PyVista backend.

    Provides tabbed visualization for different alignment stages.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.

    Signals
    -------
    view_changed
        Emitted when the camera view changes.

    Examples
    --------
    >>> viewer = Viewer3D()
    >>> viewer.set_sfm_cloud(sfm_pcd)
    >>> viewer.set_rgbd_cloud(rgbd_pcd, transform_matrix)
    """

    view_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the 3D viewer widget."""
        super().__init__(parent)
        self._setup_ui()
        self._sfm_pcd: Optional[o3d.geometry.PointCloud] = None
        self._rgbd_pcd: Optional[o3d.geometry.PointCloud] = None
        self._transform: np.ndarray = np.eye(4)

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tab widget for different views
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # Create plotter widgets for each tab
        self._plotter_raw = QtInteractor(self)
        self._plotter_aligned = QtInteractor(self)
        self._plotter_compare = QtInteractor(self)

        self._tabs.addTab(self._plotter_raw, "Raw")
        self._tabs.addTab(self._plotter_aligned, "Aligned")
        self._tabs.addTab(self._plotter_compare, "Compare")

        # Connect tab changes to update views
        self._tabs.currentChanged.connect(self._on_tab_changed)

    def _o3d_to_pyvista(
        self,
        pcd: o3d.geometry.PointCloud,
    ) -> pv.PolyData:
        """
        Convert Open3D point cloud to PyVista PolyData.

        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            Open3D point cloud.

        Returns
        -------
        pv.PolyData
            PyVista point cloud.
        """
        points = np.asarray(pcd.points)
        cloud = pv.PolyData(points)

        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
            cloud["RGB"] = colors

        return cloud

    def set_sfm_cloud(self, pcd: o3d.geometry.PointCloud) -> None:
        """
        Set the SfM point cloud for display.

        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            SfM point cloud.
        """
        self._sfm_pcd = pcd
        self._update_views()

    def set_rgbd_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set the RGBD point cloud for display.

        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            RGBD point cloud.
        transform : np.ndarray, optional
            4x4 transformation matrix to apply.
        """
        self._rgbd_pcd = pcd
        if transform is not None:
            self._transform = transform
        self._update_views()

    def set_transform(self, transform: np.ndarray) -> None:
        """
        Update the transformation matrix and refresh views.

        Parameters
        ----------
        transform : np.ndarray
            4x4 transformation matrix.
        """
        self._transform = transform
        self._update_views()

    def _update_views(self) -> None:
        """Update all visualization tabs."""
        self._update_raw_view()
        self._update_aligned_view()
        self._update_compare_view()

    def _update_raw_view(self) -> None:
        """Update the raw (untransformed) view."""
        self._plotter_raw.clear()

        if self._sfm_pcd is not None:
            cloud = self._o3d_to_pyvista(self._sfm_pcd)
            if "RGB" in cloud.array_names:
                self._plotter_raw.add_mesh(
                    cloud,
                    scalars="RGB",
                    rgb=True,
                    point_size=2,
                    render_points_as_spheres=True,
                )
            else:
                self._plotter_raw.add_mesh(
                    cloud,
                    color="blue",
                    point_size=2,
                    render_points_as_spheres=True,
                )

        if self._rgbd_pcd is not None:
            cloud = self._o3d_to_pyvista(self._rgbd_pcd)
            if "RGB" in cloud.array_names:
                self._plotter_raw.add_mesh(
                    cloud,
                    scalars="RGB",
                    rgb=True,
                    point_size=2,
                    render_points_as_spheres=True,
                )
            else:
                self._plotter_raw.add_mesh(
                    cloud,
                    color="red",
                    point_size=2,
                    render_points_as_spheres=True,
                )

        self._plotter_raw.reset_camera()

    def _update_aligned_view(self) -> None:
        """Update the aligned (transformed) view."""
        self._plotter_aligned.clear()

        if self._sfm_pcd is not None:
            cloud = self._o3d_to_pyvista(self._sfm_pcd)
            if "RGB" in cloud.array_names:
                self._plotter_aligned.add_mesh(
                    cloud,
                    scalars="RGB",
                    rgb=True,
                    point_size=2,
                    render_points_as_spheres=True,
                )
            else:
                self._plotter_aligned.add_mesh(
                    cloud,
                    color="blue",
                    point_size=2,
                    render_points_as_spheres=True,
                )

        if self._rgbd_pcd is not None:
            # Apply transformation
            import copy
            transformed = copy.deepcopy(self._rgbd_pcd)
            transformed.transform(self._transform)
            cloud = self._o3d_to_pyvista(transformed)

            if "RGB" in cloud.array_names:
                self._plotter_aligned.add_mesh(
                    cloud,
                    scalars="RGB",
                    rgb=True,
                    point_size=2,
                    render_points_as_spheres=True,
                )
            else:
                self._plotter_aligned.add_mesh(
                    cloud,
                    color="red",
                    point_size=2,
                    render_points_as_spheres=True,
                )

        self._plotter_aligned.reset_camera()

    def _update_compare_view(self) -> None:
        """Update the comparison view with offset."""
        self._plotter_compare.clear()
        offset = np.array([0.1, 0, 0])

        if self._sfm_pcd is not None:
            cloud = self._o3d_to_pyvista(self._sfm_pcd)
            cloud.points = cloud.points + offset
            if "RGB" in cloud.array_names:
                self._plotter_compare.add_mesh(
                    cloud,
                    scalars="RGB",
                    rgb=True,
                    point_size=2,
                    render_points_as_spheres=True,
                )
            else:
                self._plotter_compare.add_mesh(
                    cloud,
                    color="blue",
                    point_size=2,
                    render_points_as_spheres=True,
                )

        if self._rgbd_pcd is not None:
            import copy
            transformed = copy.deepcopy(self._rgbd_pcd)
            transformed.transform(self._transform)
            cloud = self._o3d_to_pyvista(transformed)
            cloud.points = cloud.points - offset

            if "RGB" in cloud.array_names:
                self._plotter_compare.add_mesh(
                    cloud,
                    scalars="RGB",
                    rgb=True,
                    point_size=2,
                    render_points_as_spheres=True,
                )
            else:
                self._plotter_compare.add_mesh(
                    cloud,
                    color="red",
                    point_size=2,
                    render_points_as_spheres=True,
                )

        self._plotter_compare.reset_camera()

    @Slot(int)
    def _on_tab_changed(self, index: int) -> None:
        """Handle tab change events."""
        self.view_changed.emit()

    def clear(self) -> None:
        """Clear all point clouds."""
        self._sfm_pcd = None
        self._rgbd_pcd = None
        self._transform = np.eye(4)
        self._plotter_raw.clear()
        self._plotter_aligned.clear()
        self._plotter_compare.clear()

    def close_plotters(self) -> None:
        """Clean up plotters on close."""
        self._plotter_raw.close()
        self._plotter_aligned.close()
        self._plotter_compare.close()
