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
    QLabel,
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

import io
import matplotlib
import matplotlib.pyplot as plt


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
        self._sfm_pin_data: Optional[dict] = None
        self._pin_detection_data: Optional[dict] = None
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
        self._plotter_pin_detect = QtInteractor(self)

        # SfM Pin Tab with Layout
        self._tab_sfm_pin_widget = QWidget()
        sfm_pin_layout = QVBoxLayout(self._tab_sfm_pin_widget)
        sfm_pin_layout.setContentsMargins(0, 0, 0, 0)
        
        self._plotter_sfm_pin = QtInteractor(self._tab_sfm_pin_widget)
        sfm_pin_layout.addWidget(self._plotter_sfm_pin)
        
        self._sfm_pin_colorbar_label = QLabel()
        self._sfm_pin_colorbar_label.setAlignment(Qt.AlignCenter)
        self._sfm_pin_colorbar_label.setFixedHeight(80)
        sfm_pin_layout.addWidget(self._sfm_pin_colorbar_label)

        self._tabs.addTab(self._plotter_raw, "Raw")
        self._tabs.addTab(self._tab_sfm_pin_widget, "SfM Pin")
        self._tabs.addTab(self._plotter_pin_detect, "Pin Detection")
        self._tabs.addTab(self._plotter_aligned, "Aligned")

        # Connect tab changes to update views
        self._tabs.currentChanged.connect(self._on_tab_changed)

    def _o3d_to_pyvista(
        self,
        geometry,
    ) -> pv.PolyData:
        """
        Convert Open3D geometry to PyVista PolyData.

        Parameters
        ----------
        geometry : o3d.geometry.Geometry
            Open3D geometry (PointCloud or TriangleMesh).

        Returns
        -------
        pv.PolyData
            PyVista mesh.
        """
        if isinstance(geometry, o3d.geometry.PointCloud):
            points = np.asarray(geometry.points)
            cloud = pv.PolyData(points)
            if geometry.has_colors():
                colors = (np.asarray(geometry.colors) * 255).astype(np.uint8)
                cloud["RGB"] = colors
            return cloud

        elif isinstance(geometry, o3d.geometry.TriangleMesh):
            vertices = np.asarray(geometry.vertices)
            faces = np.asarray(geometry.triangles)
            # PyVista expects faces as [n_points, p1, p2, p3, ...]
            # We assume triangles mainly, but let's be safe
            if len(faces) > 0:
                faces_with_size = np.hstack(
                    [np.full((faces.shape[0], 1), 3), faces]
                ).flatten()
                mesh = pv.PolyData(vertices, faces_with_size)
            else:
                mesh = pv.PolyData(vertices)

            if geometry.has_vertex_colors():
                colors = (np.asarray(geometry.vertex_colors) * 255).astype(np.uint8)
                mesh["RGB"] = colors
            return mesh
        
        return pv.PolyData()

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

    def _generate_colorbar_pixmap(self) -> QPixmap:
        """Generate a matplotlib colorbar for HSV distance."""
        fig, ax = plt.subplots(figsize=(10, 1))
        fig.subplots_adjust(bottom=0.7)
        
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"),
            cax=ax, orientation='horizontal'
        )
        cb.set_label('HSV color distance')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        buf.seek(0)
        plt.close(fig)
        
        qimg = QImage.fromData(buf.getvalue())
        return QPixmap.fromImage(qimg)

    def set_sfm_pin_data(self, data: dict) -> None:
        """
        Set data for SfM Pin visualization.
        
        Parameters
        ----------
        data : dict
            Dictionary with keys: 'pcd', 'pcd_offset_colormap', 'pin_pcd_strengthen'.
        """
        self._sfm_pin_data = data
        
        # Update colorbar
        pixmap = self._generate_colorbar_pixmap()
        self._sfm_pin_colorbar_label.setPixmap(pixmap)
        
        self._update_views()

    def set_pin_detection_data(self, data: dict) -> None:
        """
        Set data for Pin Detection visualization.
        
        Parameters
        ----------
        data : dict
            Dictionary containing SfM and RGBD pin detection results (disks, arrows, etc.).
        """
        self._pin_detection_data = data
        self._update_views()

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

    def _update_views(self) -> None:
        """Update all visualization tabs."""
        self._update_raw_view()
        self._update_aligned_view()
        self._update_sfm_pin_view()
        self._update_pin_detection_view()

    def _add_mesh_to_plotter(self, plotter, geometry, offset=None, color=None, point_size=2):
        """Helper to add Open3D geometry to PyVista plotter."""
        if geometry is None:
            return

        import copy
        # Work on a copy if we need to modify points (offset)
        if offset is not None:
             # Deep copy might be expensive for point clouds, handle carefully
             # But PyVista conversion does copy points anyway.
             # So we can convert then translate.
             pass

        mesh = self._o3d_to_pyvista(geometry)
        
        if offset is not None:
            mesh.points += offset
            
        kwargs = {}
        if "RGB" in mesh.array_names:
            kwargs["scalars"] = "RGB"
            kwargs["rgb"] = True
        elif color:
            kwargs["color"] = color
        
        if isinstance(geometry, o3d.geometry.PointCloud):
            kwargs["point_size"] = point_size
            kwargs["render_points_as_spheres"] = True
            
        plotter.add_mesh(mesh, **kwargs)

    def _update_sfm_pin_view(self) -> None:
        """Update the SfM Pin view."""
        self._plotter_sfm_pin.clear()
        if self._sfm_pin_data is None:
            return

        sfm_pcd = self._sfm_pin_data.get('pcd')
        if sfm_pcd is None:
            return
            
        # Calculate dynamic spacing
        bbox = sfm_pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_max_bound() - bbox.get_min_bound()
        max_extent = np.max(extent)
        spacing = max_extent * 1.25
        
        center = sfm_pcd.get_center()

        # 1. Left: Original SfM
        self._add_mesh_to_plotter(
            self._plotter_sfm_pin, 
            sfm_pcd,
            offset=np.array([-spacing, 0, 0])
        )

        # 2. Middle: HSV Colormap
        hsv_pcd = self._sfm_pin_data.get('pcd_offset_colormap')
        if hsv_pcd:
            hsv_center = hsv_pcd.get_center()
            correction = center - hsv_center
            self._add_mesh_to_plotter(
                self._plotter_sfm_pin, 
                hsv_pcd,
                offset=correction 
            )
        
        # 3. Right: Combined (HSV background + Red Pin)
        if hsv_pcd:
            hsv_center = hsv_pcd.get_center()
            correction = center - hsv_center
            self._add_mesh_to_plotter(
                self._plotter_sfm_pin, 
                hsv_pcd,
                offset=correction + np.array([spacing, 0, 0])
            )

        pin_pcd_red = self._sfm_pin_data.get('pin_pcd_strengthen')
        if pin_pcd_red:
             # Assume pin pcd (from indices) is in original sfm frame (near center)
             # So we shift it by spacing
             self._add_mesh_to_plotter(
                self._plotter_sfm_pin, 
                pin_pcd_red,
                offset=np.array([spacing, 0, 0]),
                color="red",
                point_size=5
             )
             
        self._plotter_sfm_pin.reset_camera()

    def _update_pin_detection_view(self) -> None:
        """Update the Pin Detection view."""
        self._plotter_pin_detect.clear()
        if self._pin_detection_data is None:
            return
            
        # SfM Data (Left side or Overlay?)
        # User said: "Pin Detection tab... parallel... as Figure 3"
        # Figure 3 usually implies showing results. 
        # But "SfM Pin" was tab 2 (Figure 2).
        # "Pin Detection" is tab 3.
        # Let's place SfM at Origin, RGBD at some offset (e.g. 0.2).
        
        data = self._pin_detection_data
        
        # --- SfM Group (at 0,0,0) ---
        sfm_offset = np.array([0, 0, 0])
        
        # 1. SfM Cloud (with red pin highlighting?)
        # Use 'pcd' but maybe we want to overlay 'pin_pcd' in Red?
        if 'sfm_pcd' in data:
            self._add_mesh_to_plotter(
                self._plotter_pin_detect, data['sfm_pcd'], offset=sfm_offset, color="gray"
            )
        
        if 'sfm_pin_pcd' in data:
             self._add_mesh_to_plotter(
                self._plotter_pin_detect, data['sfm_pin_pcd'], offset=sfm_offset, color="red"
            )
            
        # 2. Disk
        if 'sfm_disk' in data:
             self._add_mesh_to_plotter(
                self._plotter_pin_detect, data['sfm_disk'], offset=sfm_offset, color="red"
            )
            
        # 3. Arrow
        if 'sfm_arrow' in data:
             self._add_mesh_to_plotter(
                self._plotter_pin_detect, data['sfm_arrow'], offset=sfm_offset, color="red"
            )
            
        # --- RGBD Group (at offset) ---
        rgbd_offset = np.array([0.2, 0, 0])
        
        # 1. RGBD Cloud
        if 'rgbd_pcd' in data:
             self._add_mesh_to_plotter(
                self._plotter_pin_detect, data['rgbd_pcd'], offset=rgbd_offset, color="gray"
            )
        
        if 'rgbd_pin_pcd' in data:
             self._add_mesh_to_plotter(
                self._plotter_pin_detect, data['rgbd_pin_pcd'], offset=rgbd_offset, color="yellow"
            )

        # 2. Disk
        if 'rgbd_disk' in data:
             self._add_mesh_to_plotter(
                self._plotter_pin_detect, data['rgbd_disk'], offset=rgbd_offset, color="yellow"
            )
            
        # 3. Arrow
        if 'rgbd_arrow' in data:
             self._add_mesh_to_plotter(
                self._plotter_pin_detect, data['rgbd_arrow'], offset=rgbd_offset, color="yellow"
            )

        self._plotter_pin_detect.reset_camera()

    @Slot(int)
    def _on_tab_changed(self, index: int) -> None:
        """Handle tab change events."""
        self.view_changed.emit()

    def clear(self) -> None:
        """Clear all point clouds."""
        self._sfm_pcd = None
        self._rgbd_pcd = None
        self._sfm_pin_data = None
        self._pin_detection_data = None
        self._transform = np.eye(4)
        self._plotter_raw.clear()
        self._plotter_aligned.clear()
        self._plotter_sfm_pin.clear()
        self._plotter_pin_detect.clear()

    def close_plotters(self) -> None:
        """Clean up plotters on close."""
        self._plotter_raw.close()
        self._plotter_aligned.close()
        self._plotter_sfm_pin.close()
        self._plotter_pin_detect.close()
