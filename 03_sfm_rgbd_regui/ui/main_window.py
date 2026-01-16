# -*- coding: utf-8 -*-
"""
Main application window for SFM-RGBD registration GUI.

Integrates all widgets and manages the registration workflow.
"""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Slot, QSettings
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    AlignmentParams,
    Aligner,
    DataConfig,
    DataLoader,
    save_result_json,
)
from widgets import FileTreeWidget, ParameterPanel, RmseChartWidget, Viewer3D
from ui.preferences_dialog import PreferencesDialog
from utils import pin_center as util_pc
from loguru import logger


class MainWindow(QMainWindow):
    """
    Main application window for the registration GUI.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.

    Examples
    --------
    >>> app = QApplication([])
    >>> window = MainWindow()
    >>> window.show()
    >>> app.exec()
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the main window."""
        super().__init__(parent)
        self.setWindowTitle("SFM-RGBD Registration Tool")
        self.setMinimumSize(1200, 800)

        # State
        self._config: Optional[DataConfig] = None
        self._loader: Optional[DataLoader] = None
        self._aligner: Optional[Aligner] = None
        self._current_pid: Optional[str] = None
        self._current_rgbd: Optional[dict] = None
        self._current_sfm: Optional[dict] = None
        self._current_result = None
        self._settings = QSettings("PotatoScan", "RegistrationGUI")

        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._load_settings()

    def _setup_ui(self) -> None:
        """Set up the main window UI."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(8)

        # Left panel
        left_panel = self._create_left_panel()

        # Right panel (viewer + chart)
        right_panel = self._create_right_panel()

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])

        main_layout.addWidget(splitter)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready. Select folders to begin.")

    def _create_left_panel(self) -> QWidget:
        """Create the left panel with folders, file tree, and parameters."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        # Folder selection group
        folder_group = QGroupBox("Dataset")
        folder_layout = QVBoxLayout(folder_group)

        # Dataset root
        self._lbl_dataset = QLabel("Dataset: Not selected")
        self._lbl_dataset.setWordWrap(True)
        self._btn_dataset = QPushButton("Select Dataset Folder...")
        self._btn_dataset.clicked.connect(self._on_select_dataset)
        folder_layout.addWidget(self._lbl_dataset)
        folder_layout.addWidget(self._btn_dataset)

        layout.addWidget(folder_group)

        # File tree
        tree_group = QGroupBox("Files")
        tree_layout = QVBoxLayout(tree_group)
        self._file_tree = FileTreeWidget()
        self._file_tree.item_selected.connect(self._on_item_selected)
        tree_layout.addWidget(self._file_tree)
        layout.addWidget(tree_group)

        # Parameters
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_group)
        self._param_panel = ParameterPanel()
        self._param_panel.params_changed.connect(self._on_params_changed)
        param_layout.addWidget(self._param_panel)
        layout.addWidget(param_group)

        # Navigation buttons (Moved from right panel)
        nav_layout = QHBoxLayout()
        # nav_layout.addStretch() # Don't need stretch here if we want them to expand or fill

        self._btn_prev = QPushButton("< Previous")
        self._btn_prev.clicked.connect(self._on_prev_item)
        nav_layout.addWidget(self._btn_prev)

        self._btn_save = QPushButton("Save && Next >")
        self._btn_save.clicked.connect(self._on_save_and_next)
        self._btn_save.setDefault(True)
        nav_layout.addWidget(self._btn_save)

        layout.addLayout(nav_layout)

        return panel

    def _create_right_panel(self) -> QWidget:
        """Create the right panel with 3D viewer and RMSE chart."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)

        # 3D Viewer
        self._viewer = Viewer3D()
        layout.addWidget(self._viewer, stretch=3)

        # RMSE Chart
        chart_group = QGroupBox("RMSE Analysis")
        chart_layout = QVBoxLayout(chart_group)
        self._rmse_chart = RmseChartWidget()
        self._rmse_chart.peak_changed.connect(self._on_peak_changed)
        chart_layout.addWidget(self._rmse_chart)
        # Increased stretch factor to give more space
        layout.addWidget(chart_group, stretch=2)

        return panel

    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Dataset...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._on_select_dataset)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        prefs_action = QAction("&Preferences...", self)
        prefs_action.setShortcut("Ctrl+,")
        prefs_action.triggered.connect(self._on_open_preferences)
        edit_menu.addAction(prefs_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        refresh_action = QAction("&Refresh", self)
        refresh_action.setShortcut(QKeySequence.Refresh)
        refresh_action.triggered.connect(self._refresh_file_list)
        view_menu.addAction(refresh_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_shortcuts(self) -> None:
        """Set up keyboard shortcuts."""
        self._shortcuts: dict[str, QShortcut] = {}

        shortcuts_config = self._settings.value(
            "shortcuts",
            PreferencesDialog.DEFAULT_SHORTCUTS,
        )

        for action_id, default in PreferencesDialog.DEFAULT_SHORTCUTS.items():
            key = shortcuts_config.get(action_id, default)
            shortcut = QShortcut(QKeySequence(key), self)
            self._shortcuts[action_id] = shortcut

        # Connect shortcuts
        self._shortcuts["prev_item"].activated.connect(self._on_prev_item)
        self._shortcuts["next_item"].activated.connect(self._on_next_item)
        self._shortcuts["save_current"].activated.connect(self._on_save_current)
        self._shortcuts["save_and_next"].activated.connect(self._on_save_and_next)
        self._shortcuts["prev_peak"].activated.connect(self._rmse_chart._on_prev_peak)
        self._shortcuts["next_peak"].activated.connect(self._rmse_chart._on_next_peak)

    def _load_settings(self) -> None:
        """Load saved settings."""
        # Window geometry
        geometry = self._settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Developer mode (log level)
        dev_mode = self._settings.value("developer_mode", False, type=bool)
        self._apply_log_level(dev_mode)

        # Last used folders
        dataset = self._settings.value("last_dataset")

        if dataset and Path(dataset).exists():
            self._lbl_dataset.setText(f"Dataset: {dataset}")
            self._try_init_loader()

    def _save_settings(self) -> None:
        """Save current settings."""
        self._settings.setValue("geometry", self.saveGeometry())

    @Slot()
    def _on_select_dataset(self) -> None:
        """Handle dataset folder selection."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Root Folder",
            str(Path.home()),
        )
        if folder:
            self._lbl_dataset.setText(f"Dataset: {folder}")
            self._settings.setValue("last_dataset", folder)
            self._try_init_loader()

    def _try_init_loader(self) -> None:
        """Try to initialize the data loader if dataset folder is set."""
        dataset = self._settings.value("last_dataset")

        if not dataset:
            return
        if not Path(dataset).exists():
            return

        self._config = DataConfig(dataset_root=Path(dataset))
        self._loader = DataLoader(self._config)
        self._aligner = Aligner(
            params=self._param_panel.get_params(),
            on_update=self._on_alignment_update,
        )

        self._refresh_file_list()
        self._status_bar.showMessage("Dataset loaded. Select an item to process.")

    def _refresh_file_list(self) -> None:
        """Refresh the file list."""
        if self._loader is None:
            return

        ids = self._loader.get_ids()
        completed = self._loader.get_completed_ids()
        pin_colors = self._loader.get_pin_colors()
        self._file_tree.set_items(ids, completed, pin_colors)

    @Slot(str)
    def _on_item_selected(self, pid: str) -> None:
        """Handle item selection in file tree."""
        if self._loader is None:
            return

        self._current_pid = pid
        self._status_bar.showMessage(f"Loading {pid}...")
        QApplication.processEvents()

        try:
            # Load data
            self._current_rgbd = self._loader.load_rgbd(pid, visualize=True)
            self._current_sfm = self._loader.load_sfm(
                pid,
                visualize=True,
                status_callback=self._update_statusbar,
            )

            # Set point clouds in viewer
            self._viewer.set_sfm_cloud(self._current_sfm["pcd"])
            self._viewer.set_rgbd_cloud(self._current_rgbd["pcd"])
            
            # --- 1. SfM Pin Tab Data ---
            # Pass the data loaded by SfMPinFetcher directly to the viewer
            # It already contains 'pcd', 'pcd_offset_colormap', 'pin_pcd_strengthen'
            self._viewer.set_sfm_pin_data(self._current_sfm)

            # --- 2. Pin Detection Tab Data ---
            # Calculate pin center geometry (disk, arrow) for SfM
            sfm_pin_result = util_pc.find_pin_center(
                self._current_sfm['pin_pcd'], 
                self._current_sfm['pcd'], 
                circle_color=[1, 0, 0], # Red
                visualize=True, 
                show=False, 
                label="sfm"
            )
            
            # Calculate pin center geometry for RGBD
            rgbd_pin_result = util_pc.find_pin_center(
                self._current_rgbd['pin_pcd'], 
                self._current_rgbd['pcd'], 
                circle_color=[1, 1, 0], # Yellow
                visualize=True, 
                show=False, 
                label="rgbd"
            )
            
            # Combine into one dictionary for the viewer
            pin_detect_data = {
                # SfM Group
                'sfm_pcd': self._current_sfm['pcd'],
                'sfm_pin_pcd': self._current_sfm['pin_pcd'], # Or 'pin_pcd_strengthen' if available/preferred
                'sfm_disk': sfm_pin_result.get('circle_mesh'),
                'sfm_arrow': sfm_pin_result.get('vector_arrow'),
                
                # RGBD Group
                'rgbd_pcd': self._current_rgbd['pcd'],
                'rgbd_pin_pcd': self._current_rgbd['pin_pcd'],
                'rgbd_disk': rgbd_pin_result.get('circle_mesh'),
                'rgbd_arrow': rgbd_pin_result.get('vector_arrow'),
            }
            
            self._viewer.set_pin_detection_data(pin_detect_data)

            # Run alignment
            self._run_alignment()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load {pid}:\n{e}")
            self._status_bar.showMessage(f"Error loading {pid}")

    def _run_alignment(self) -> None:
        """Run the alignment pipeline."""
        if self._aligner is None or self._current_rgbd is None:
            logger.warning("Aligner or RGBD data missing")
            return

        self._status_bar.showMessage(f"Aligning {self._current_pid}...")
        logger.info(f"Starting alignment for {self._current_pid}")
        QApplication.processEvents()

        try:
            params = self._param_panel.get_params()
            logger.debug(f"Alignment params: {params}")
            self._aligner.update_params(params)
            
            logger.info("Computing full alignment...")
            self._current_result = self._aligner.compute_full_alignment(
                self._current_rgbd,
                self._current_sfm,
            )
            logger.success(f"Alignment complete. RMSE: {self._current_result.rmse}")

            # Update views
            self._viewer.set_transform(self._current_result.transform_matrix)

            # Update chart
            angles, rmses = self._current_result.rmse_curve
            # Get peak indices from peak angles
            peak_indices = []
            for pa in self._current_result.peak_angles:
                idx = int(pa / 10) - 1
                if 0 <= idx < len(angles):
                    peak_indices.append(idx)
                    
            logger.debug(f"Updating chart with {len(angles)} points and {len(peak_indices)} peaks")
            
            self._rmse_chart.set_data(
                angles,
                rmses,
                peak_indices,
                self._current_result.selected_peak_idx,
            )

            self._status_bar.showMessage(
                f"{self._current_pid}: RMSE={self._current_result.rmse:.6f}"
            )

        except Exception as e:
            logger.exception(f"Alignment failed for {self._current_pid}")
            QMessageBox.warning(self, "Alignment Error", str(e))
            self._status_bar.showMessage("Alignment failed")

    def _on_alignment_update(self, stage: str, data: dict) -> None:
        """Handle alignment progress updates."""
        self._status_bar.showMessage(f"{stage}: {data.get('status', '')}")
        QApplication.processEvents()

    def _update_statusbar(self, message: str) -> None:
        """
        Update the statusbar with a message and process events.
        
        This method can be used as a callback for long-running operations
        to provide real-time feedback in the GUI.
        
        Parameters
        ----------
        message : str
            The message to display in the statusbar.
        """
        self._status_bar.showMessage(message)
        QApplication.processEvents()

    @Slot(object)
    def _on_params_changed(self, params: AlignmentParams) -> None:
        """Handle parameter changes for real-time update."""
        if self._current_result is None:
            return

        # Recompute ICP with new parameters
        if self._aligner is not None:
            self._aligner.update_params(params)
            self._run_alignment()

    @Slot(int)
    def _on_peak_changed(self, peak_idx: int) -> None:
        """Handle peak selection change."""
        if self._aligner is None or self._current_rgbd is None:
            return

        self._current_result = self._aligner.recompute_with_peak(
            peak_idx,
            self._current_rgbd,
            self._current_sfm,
        )
        self._viewer.set_transform(self._current_result.transform_matrix)
        self._status_bar.showMessage(
            f"{self._current_pid}: RMSE={self._current_result.rmse:.6f} (peak {peak_idx})"
        )

    @Slot()
    def _on_prev_item(self) -> None:
        """Navigate to previous item."""
        # TODO: Implement previous item navigation
        pass

    @Slot()
    def _on_next_item(self) -> None:
        """Navigate to next item."""
        self._file_tree.select_next_uncompleted()

    @Slot()
    def _on_save_current(self) -> None:
        """Save current alignment result."""
        self._save_result()

    @Slot()
    def _on_save_and_next(self) -> None:
        """Save current result and move to next item."""
        if self._save_result():
            self._file_tree.select_next_uncompleted()

    def _save_result(self) -> bool:
        """Save the current alignment result to JSON."""
        if self._current_result is None or self._current_pid is None:
            QMessageBox.warning(self, "Warning", "No alignment result to save.")
            return False

        if self._config is None:
            return False

        try:
            output_path = self._loader.get_output_path(self._current_pid)
            params = self._param_panel.get_params()

            save_result_json(
                output_path=output_path,
                rgbd_pcd_file=self._current_rgbd.get("pcd_rela_path", ""),
                sfm_mesh_file=self._current_sfm.get("pcd_rela_path", ""),
                transform_matrix=self._current_result.transform_matrix,
                rmse=self._current_result.rmse,
                open3d_rmse=self._current_result.open3d_rmse,
                sfm_pin_data=self._current_result.sfm_pin_data,
                rgbd_pin_data=self._current_result.rgbd_pin_data,
                search_radius=params.search_radius,
                cross_buffer=params.cross_buffer,
                icp_iter_num=params.icp_iter_num,
                icp_threshold=params.icp_threshold,
                geometry_weight=params.geometry_weight,
                hsv_weight=self._current_sfm.get("hsv_weight"),
                hsv_denoise_threshold=self._current_sfm.get("stop_thresh"),
                hsv_denoised_volume=self._current_sfm.get("stop_hull_volume"),
            )

            self._file_tree.set_completed(self._current_pid, True)
            self._status_bar.showMessage(f"Saved {self._current_pid}")
            return True

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")
            return False

    @Slot()
    def _on_open_preferences(self) -> None:
        """Open the preferences dialog."""
        dialog = PreferencesDialog(self)
        dialog.set_shortcuts(
            self._settings.value(
                "shortcuts",
                PreferencesDialog.DEFAULT_SHORTCUTS,
            )
        )
        dialog.set_developer_mode(
            self._settings.value("developer_mode", False, type=bool)
        )

        if dialog.exec():
            # Save and apply shortcuts
            shortcuts = dialog.get_shortcuts()
            self._settings.setValue("shortcuts", shortcuts)
            for action_id, key in shortcuts.items():
                if action_id in self._shortcuts:
                    self._shortcuts[action_id].setKey(QKeySequence(key))

            # Save and apply developer mode
            dev_mode = dialog.get_developer_mode()
            self._settings.setValue("developer_mode", dev_mode)
            self._apply_log_level(dev_mode)

    def _apply_log_level(self, debug_enabled: bool) -> None:
        """
        Apply the log level based on developer mode setting.

        Parameters
        ----------
        debug_enabled : bool
            True to enable DEBUG level, False for INFO level.
        """
        import sys
        logger.remove()
        level = "DEBUG" if debug_enabled else "INFO"
        logger.add(sys.stderr, level=level)
        logger.info(f"Log level set to {level}")

    @Slot()
    def _on_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About SFM-RGBD Registration Tool",
            "SFM-RGBD Registration Tool v0.1.0\n\n"
            "Interactive GUI for point cloud registration\n"
            "using pin-based alignment and ICP refinement.",
        )

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        self._save_settings()
        self._viewer.close_plotters()
        super().closeEvent(event)
