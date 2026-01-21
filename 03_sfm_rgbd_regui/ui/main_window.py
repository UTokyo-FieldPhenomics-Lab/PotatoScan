# -*- coding: utf-8 -*-
"""
Main application window for SFM-RGBD registration GUI.

Integrates all widgets and manages the registration workflow.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, Slot, QSettings
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
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
    load_result_json,
)
from widgets import FileTreeWidget, ParameterPanel, RmseChartWidget, Viewer3D
from ui.preferences_dialog import PreferencesDialog
from utils import pin_center as util_pc
from utils.pin_segment import InsufficientPinPointsError
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
        self._is_dirty: bool = False
        self._manual_specified_angles: list[int] = []  # Manual rotation angles
        self._current_selected_angle: Optional[int] = None  # Track actively selected angle
        self._invert_sfm_vector: bool = False
        self._invert_rgbd_vector: bool = False
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

        # Connect compare mode signal (now that _viewer exists)
        self._param_panel.compare_mode_changed.connect(
            self._viewer.set_compare_mode
        )

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
        self._file_tree.change_rgbd_requested.connect(self._on_change_rgbd_cloud_requested)
        tree_layout.addWidget(self._file_tree)
        layout.addWidget(tree_group)

        # Parameters
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_group)
        self._param_panel = ParameterPanel()
        self._param_panel.params_changed.connect(self._on_params_changed)
        self._param_panel.sfm_pin_params_changed.connect(
            self._on_sfm_pin_params_changed
        )
        # Note: compare_mode_changed is connected in _setup_ui after _viewer is created
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

        # Reset parameters action
        reset_params_action = QAction("&Reset Parameters", self)
        reset_params_action.triggered.connect(self._on_reset_parameters)
        edit_menu.addAction(reset_params_action)

        edit_menu.addSeparator()

        # Rotation Angle submenu
        rotation_menu = QMenu("Rotation &Angle", self)
        edit_menu.addMenu(rotation_menu)

        manual_angle_action = QAction("&Manual Specify...", self)
        manual_angle_action.triggered.connect(self._on_manual_angle_specify)
        rotation_menu.addAction(manual_angle_action)

        reset_angles_action = QAction("&Reset Manual Angles", self)
        reset_angles_action.triggered.connect(self._on_reset_manual_angles)
        rotation_menu.addAction(reset_angles_action)

        edit_menu.addSeparator()

        # Pin Vector submenu
        pin_vector_menu = QMenu("&Pin Vector", self)
        edit_menu.addMenu(pin_vector_menu)

        self._action_invert_sfm = QAction("Invert SfM Vector", self)
        self._action_invert_sfm.setCheckable(True)
        self._action_invert_sfm.triggered.connect(self._on_invert_sfm_vector)
        pin_vector_menu.addAction(self._action_invert_sfm)

        self._action_invert_rgbd = QAction("Invert RGBD Vector", self)
        self._action_invert_rgbd.setCheckable(True)
        self._action_invert_rgbd.triggered.connect(self._on_invert_rgbd_vector)
        pin_vector_menu.addAction(self._action_invert_rgbd)

        edit_menu.addSeparator()

        change_rgbd_action = QAction("Change &RGBD Point Cloud...", self)
        change_rgbd_action.setStatusTip("Select a different RGBD point cloud file for the current item")
        change_rgbd_action.triggered.connect(self._on_change_rgbd_menu_action)
        edit_menu.addAction(change_rgbd_action)

        edit_menu.addSeparator()

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

        # Point size modifier for 3D viewer
        modifier_name = self._settings.value(
            "point_size_modifier",
            PreferencesDialog.DEFAULT_MODIFIER,
        )
        self._apply_point_size_modifier(modifier_name)

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
        
        # Get saved metadata for RGBD files
        saved_meta = self._loader.get_saved_metadata()
        
        # Get default files for all IDs
        defaults = self._loader.get_all_default_images()
        logger.debug(f"Loaded {len(defaults)} default RGBD files")
        logger.debug(f"Saved metadata entries: {len(saved_meta)}")
        
        # Combine: Start with defaults, override with saved
        combined_files = defaults.copy()
        for pid, meta in saved_meta.items():
            if meta.get('rgbd_filename'):
                combined_files[pid] = meta['rgbd_filename']
        
        self._file_tree.set_items(ids, completed, pin_colors, combined_files)
        
        # Now apply manual indicators for saved items
        # This is efficient enough? Or should we modify set_items?
        # Iterating set_items is O(N).
        # We can iterate completed items and check if they differ from default or just trust saved_meta implies manual logic?
        # Actually, if a result IS saved, we trust that file is the intended one.
        # Visual indication of "manual" might be useful if it differs from default.
        # For now, let's just make sure the column is populated.
        
        for pid, meta in saved_meta.items():
            fname = meta.get('rgbd_filename')
            if fname:
                # Check if it differs from default
                default_fname = defaults.get(pid)
                is_manual = (fname != default_fname)
                self._file_tree.update_rgbd_file(pid, fname, is_manual=is_manual)

    def _check_unsaved_changes(self) -> bool:
        """
        Check for unsaved changes and prompt user.

        Returns
        -------
        bool
            True if it's safe to proceed (Saved, Discarded, or No changes).
            False if cancelled.
        """
        if not self._is_dirty:
            return True

        if not self._current_pid:
            return True

        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            f"Changes to {self._current_pid} have not been saved.\nSave before continuing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )

        if reply == QMessageBox.Save:
            return self._save_result()
        elif reply == QMessageBox.Discard:
            return True
        else:
            return False

    @Slot(str)
    def _on_change_rgbd_cloud_requested(self, pid: str) -> None:
        """
        Handle request to change RGBD cloud for a potato ID.
        
        Parameters
        ----------
        pid : str
            Potato ID.
        """
        if self._loader is None:
            return
            
        # Get available files
        available_files = self._loader.get_available_rgbd_files(pid)
        if not available_files:
            QMessageBox.warning(self, "No Files", f"No RGBD files found for {pid}")
            return
            
        # Create list of filenames for selection
        # Format: "filename (pos XX)"
        items = []
        file_map = {}
        current_selection = 0
        
        # Check current loaded file to set default selection
        current_pcd_path = None
        if  self._current_rgbd and self._current_pid == pid:
             current_pcd_path = self._current_rgbd.get('pcd_rela_path')
             # pcd_rela_path: "1_rgbd/2_pcd/2025-015_pcd_365.ply" or similar
             # We need just the filename part for comparison usually, or careful matching
             
        for i, data in enumerate(available_files):
            fname = data['pcd_filename']
            pos = data['pos']
            label = f"{fname} (Pos: {pos})"
            items.append(label)
            file_map[label] = fname
            
            # Simple matching logic
            if current_pcd_path and fname in current_pcd_path:
                current_selection = i
                
        # Show dialog
        item, ok = QInputDialog.getItem(
            self, 
            "Select RGBD Point Cloud", 
            f"Select RGBD point cloud for {pid}:", 
            items, 
            current_selection, 
            False
        )
        
        if ok and item:
            selected_filename = file_map[item]
            logger.info(f"User selected RGBD file: {selected_filename} for {pid}")
            
            # Save this manually selected file temporarily until actual save
            # But we should reload immediately
            
            # If current item is this PID, reload it
            if self._current_pid == pid:
                # We need to reload the item with this specific file
                # We can call load_rgbd directly then run alignment
                
                self._status_bar.showMessage(f"Loading {selected_filename}...")
                QApplication.processEvents()
                
                try:
                    # Load new RGBD
                    self._current_rgbd = self._loader.load_rgbd(
                        pid, 
                        img_id=selected_filename, 
                        visualize=True
                    )
                     
                     # Update viewer
                    self._viewer.set_raw_cloud(
                        self._current_sfm["pcd"], self._current_rgbd["pcd"]
                    )
                    self._viewer.set_rgbd_data(self._current_rgbd)
                    
                    # Store info for saving later
                    # (We rely on _current_rgbd containing the path)
                    
                    # Re-run alignment logic (similar to on_item_selected but lighter)
                    sfm_pin_result = util_pc.find_pin_center(
                        self._current_sfm['pin_pcd'], self._current_sfm['pcd'], 
                        circle_color=[0, 0, 0], visualize=True, show=False, label="sfm"
                    )
                    rgbd_pin_result = util_pc.find_pin_center(
                        self._current_rgbd['pin_pcd'], self._current_rgbd['pcd'], 
                        circle_color=[0, 0, 0], visualize=True, show=False, label="rgbd"
                    )
                    
                    pin_detect_data = {
                        'sfm_pcd': self._current_sfm['pcd'],
                        'sfm_pin_pcd': self._current_sfm['pin_pcd'],
                        'sfm_disk': sfm_pin_result.get('circle_mesh'),
                        'sfm_arrow': sfm_pin_result.get('vector_arrow'),
                        'rgbd_pcd': self._current_rgbd['pcd'],
                        'rgbd_pin_pcd': self._current_rgbd['pin_pcd'],
                        'rgbd_disk': rgbd_pin_result.get('circle_mesh'),
                        'rgbd_arrow': rgbd_pin_result.get('vector_arrow'),
                    }
                    self._viewer.set_pin_detection_data(pin_detect_data)
                    
                    # Re-run alignment
                    self._run_alignment(selected_peak_angle=self._current_selected_angle, is_dirty=True)
                    
                    # Valid manual selection - check if different from default
                    defaults = self._loader.get_all_default_images()
                    default_filename = defaults.get(pid)
                    is_manual = (selected_filename != default_filename)
                    
                    self._file_tree.update_rgbd_file(pid, selected_filename, is_manual=is_manual)
                    
                except Exception as e:
                    logger.exception("Failed to load selected RGBD file")
                    QMessageBox.critical(self, "Error", f"Failed to load {selected_filename}:\n{e}")

    @Slot()
    def _on_change_rgbd_menu_action(self) -> None:
        """Handle Change RGBD Point Cloud action from main menu."""
        if not self._current_pid:
            QMessageBox.information(self, "No Selection", "Please select an item first.")
            return
        self._on_change_rgbd_cloud_requested(self._current_pid)

    @Slot()
    def _on_invert_sfm_vector(self) -> None:
        """Handle Invert SfM Vector action."""
        if not self._current_pid:
             # Revert check state if no items selected
            self._action_invert_sfm.setChecked(False)
            return
            
        self._invert_sfm_vector = self._action_invert_sfm.isChecked()
        logger.info(f"User toggled SfM vector inversion: {self._invert_sfm_vector}")
        self._run_alignment(selected_peak_angle=self._current_selected_angle, is_dirty=True)

    @Slot()
    def _on_invert_rgbd_vector(self) -> None:
        """Handle Invert RGBD Vector action."""
        if not self._current_pid:
            # Revert check state if no items selected
            self._action_invert_rgbd.setChecked(False)
            return

        self._invert_rgbd_vector = self._action_invert_rgbd.isChecked()
        logger.info(f"User toggled RGBD vector inversion: {self._invert_rgbd_vector}")
        self._run_alignment(selected_peak_angle=self._current_selected_angle, is_dirty=True)

    def _save_result(self) -> bool:
        """
        Save the current alignment result to JSON.

        Returns
        -------
        bool
            True if successful.
        """
        if self._current_result is None or self._current_pid is None:
            return False

        try:
            output_path = self._loader.get_output_path(self._current_pid)
            
            # Extract paths
            rgbd_pcd_file = self._current_rgbd.get("pcd_rela_path", "")
            sfm_mesh_file = self._current_sfm.get("pcd_rela_path", "").replace("2_pcd", "1_mesh").replace(".ply", ".obj")
            
            sfm_pin_params = self._param_panel.get_sfm_pin_params()
            params = self._param_panel.get_params()
            
            save_result_json(
                output_path,
                rgbd_pcd_file,
                sfm_mesh_file,
                self._current_result.transform_matrix,
                rmse=self._current_result.rmse,
                open3d_rmse=self._current_result.open3d_rmse,
                sfm_pin_data=self._current_result.sfm_pin_data,
                rgbd_pin_data=self._current_result.rgbd_pin_data,
                search_radius=params.search_radius,
                cross_buffer=params.cross_buffer,
                icp_iter_num=params.icp_iter_num,
                icp_threshold=params.icp_threshold,
                geometry_weight=params.geometry_weight,
                hsv_weight=sfm_pin_params.hsv_weights,
                peak_angles=self._current_result.peak_angles,
                selected_peak_idx=self._current_result.selected_peak_idx,
                manual_potential_indices=self._current_result.manual_potential_indices,
            )

            self._is_dirty = False
            self._loader.get_completed_ids() # Refresh cache if needed?
            self._file_tree.set_completed(self._current_pid, True)
            self._status_bar.showMessage(f"Saved {self._current_pid}")
            return True

        except Exception as e:
            logger.exception(f"Failed to save result for {self._current_pid}")
            QMessageBox.critical(self, "Save Error", str(e))
            return False

    @Slot(str)
    def _on_item_selected(self, pid: str) -> None:
        """Handle item selection in file tree."""
        if self._loader is None:
            return

        # Check for unsaved changes
        if not self._check_unsaved_changes():
            # Block selection change? Use the file tree logic if possible
            # But the signal is already emitted. We might need to handle this better.
            # For now, we proceed, but real app might revert selection.
            # If we return here, the UI selected item is different from internal state.
            
            # Revert selection in tree to match current_pid
            if self._current_pid:
                self._file_tree.select_item(self._current_pid)
            return

        self._current_pid = pid
        self._manual_specified_angles = []  # Clear manual angles for new item
        self._current_selected_angle = None  # Reset selected angle
        
        # Reset inversion flags
        self._invert_sfm_vector = False
        self._invert_rgbd_vector = False
        self._action_invert_sfm.setChecked(False)
        self._action_invert_rgbd.setChecked(False)

        # Reset ALL Steps (2, 3, 4) for new item
        self._param_panel.reset_all_steps()

        # Clear viewer and chart for fresh state
        self._viewer.clear()
        self._rmse_chart.clear()

        self._status_bar.showMessage(f"Loading {pid}...")
        QApplication.processEvents()

        try:
            # Check for existing result to determine which RGBD file to load
            output_path = self._loader.get_output_path(pid)
            rgbd_file_to_load = None
            is_manual_rgbd = False
            
            # Temporary store for loaded inversion flags if JSON exists
            loaded_invert_sfm = False
            loaded_invert_rgbd = False
            
            if output_path.exists():
                try:
                    result_json = load_result_json(output_path)
                    # Get persistent RGBD file path
                    saved_rgbd_file = result_json.get("rgbd_pcd_file", "")
                    if saved_rgbd_file:
                        # Extract basename from relative path e.g. "1_rgbd/2_pcd/2025-015_pcd_365.ply" -> "2025-015_pcd_365.ply"
                        rgbd_file_to_load = saved_rgbd_file.split("/")[-1]
                        is_manual_rgbd = True # Assume if it's saved, we respect it as "manual" or "fixed"
                    
                    # Check for inversion flags in metadata
                    meta = result_json.get("meta", {})
                    pin_seg = meta.get("pin_segment", {})
                    
                    # Check SfM
                    sfm_meta = pin_seg.get("sfm", {})
                    # Load "true" string or boolean True
                    sfm_val = sfm_meta.get("normal_vector_invert", False)
                    if isinstance(sfm_val, str) and sfm_val.lower() == "true":
                        loaded_invert_sfm = True
                    elif isinstance(sfm_val, bool) and sfm_val:
                        loaded_invert_sfm = True
                        
                    # Check RGBD
                    rgbd_meta = pin_seg.get("rgbd", {})
                    rgbd_val = rgbd_meta.get("normal_vector_invert", False)
                    if isinstance(rgbd_val, str) and rgbd_val.lower() == "true":
                        loaded_invert_rgbd = True
                    elif isinstance(rgbd_val, bool) and rgbd_val:
                        loaded_invert_rgbd = True
                        
                except Exception as e:
                    logger.warning(f"Could not read existing JSON for RGBD file or flags: {e}")

            # Apply loaded flags
            self._invert_sfm_vector = loaded_invert_sfm
            self._invert_rgbd_vector = loaded_invert_rgbd
            self._action_invert_sfm.setChecked(loaded_invert_sfm)
            self._action_invert_rgbd.setChecked(loaded_invert_rgbd)

            # Get current SfM pin params for loading
            sfm_params = self._param_panel.get_sfm_pin_params()

            # Load data
            # Use specific file if found in JSON, otherwise None (defaults to center)
            self._current_rgbd = self._loader.load_rgbd(pid, img_id=rgbd_file_to_load, visualize=True)
            
            # Update File Tree Column
            loaded_path = self._current_rgbd.get("pcd_rela_path", "")
            loaded_filename = loaded_path.split("/")[-1]
            
            # Check if this differs from default to decide on "manual" indicator
            defaults = self._loader.get_all_default_images()
            default_filename = defaults.get(pid)
            
            # If loaded from JSON or manually selected, we check difference
            # If checks pass, is_manual_rgbd becomes true only if different from default
            if loaded_filename != default_filename:
                is_manual_rgbd = True
            else:
                is_manual_rgbd = False
                
            self._file_tree.update_rgbd_file(pid, loaded_filename, is_manual=is_manual_rgbd)

            self._current_sfm = self._loader.load_sfm(
                pid,
                visualize=True,
                status_callback=self._update_statusbar,
                initial_thresh=sfm_params.initial_threshold,
                hsv_weights=sfm_params.hsv_weights,
                target_hull_volume=sfm_params.target_hull_volume,
                threshold_callback=self._on_threshold_update,
                auto_iteration=sfm_params.auto_iteration,
            )


            logger.info("Loaded SfM data: {}", self._current_sfm)
            logger.info("Loaded RGBD data: {}", self._current_rgbd)

            # Set point clouds in viewer
            self._viewer.set_raw_cloud(self._current_sfm["pcd"], self._current_rgbd["pcd"])
            
            # --- 1. SfM Pin Tab Data ---
            # Pass the data loaded by SfMPinFetcher directly to the viewer
            # It already contains 'pcd', 'pcd_offset_colormap', 'pin_pcd_strengthen'
            self._viewer.set_sfm_pin_data(self._current_sfm)
            
            # Store RGBD data for compare mode colorization
            self._viewer.set_rgbd_data(self._current_rgbd)

            # --- 2. Pin Detection Tab Data ---
            # Calculate pin center geometry (disk, arrow) for SfM
            sfm_pin_result = util_pc.find_pin_center(
                self._current_sfm['pin_pcd'], 
                self._current_sfm['pcd'], 
                circle_color=[0, 0, 0], # Black
                visualize=True, 
                show=False, 
                label="sfm"
            )
            
            # Calculate pin center geometry for RGBD
            rgbd_pin_result = util_pc.find_pin_center(
                self._current_rgbd['pin_pcd'], 
                self._current_rgbd['pcd'], 
                circle_color=[0, 0, 0], # Black
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

            # Check for existing result and load parameters
            output_path = self._loader.get_output_path(pid)
            selected_peak_angle = None  # Default to auto-select best peak
            
            if output_path.exists():
                try:
                    result_json = load_result_json(output_path)
                    meta = result_json.get("meta", {})
                    
                    pin_nbr = meta.get("pin_neighbor", {})
                    icp_cfg = meta.get("class_based_icp", {})
                    
                    # Create parameters from JSON (respecting typo in key)
                    saved_params = AlignmentParams(
                        search_radius=pin_nbr.get("search_radius(m)", 0.03),
                        cross_buffer=pin_nbr.get("corss_buffer(m)", 0.001),
                        icp_threshold=icp_cfg.get("iter_distance(m)", 0.001),
                        icp_iter_num=icp_cfg.get("iter_num", 0),
                        geometry_weight=icp_cfg.get("geometry_weight", 0.1),
                    )
                    
                    # Update UI and Aligner
                    self._param_panel.set_params(saved_params)
                    self._aligner.update_params(saved_params)

                    # Retrieve selected peak angle from saved data
                    # Old format stores index into RMSE-sorted array, so we need the actual angle
                    rms_analysis = meta.get("rms_analysis", {})
                    saved_peak_angles = rms_analysis.get("potential_local_minima", [])
                    saved_selected_idx = rms_analysis.get("selected", 0)
                    
                    # Get the actual angle value that was selected
                    if saved_peak_angles and saved_selected_idx < len(saved_peak_angles):
                        selected_peak_angle = saved_peak_angles[saved_selected_idx]
                        logger.info(
                            f"Restored selected peak angle: {selected_peak_angle}° "
                            f"(index {saved_selected_idx} in saved file)"
                        )
                    else:
                        selected_peak_angle = None
                    
                    # Restore manual angles from JSON
                    manual_potential = rms_analysis.get("manual_potential", [])
                    if manual_potential:
                        self._manual_specified_angles = [
                            saved_peak_angles[i] for i in manual_potential
                            if i < len(saved_peak_angles)
                        ]
                        logger.info(
                            f"Restored {len(self._manual_specified_angles)} "
                            f"manual angles: {self._manual_specified_angles}"
                        )
                    else:
                        self._manual_specified_angles = []
                    
                    logger.info(f"Restored parameters from {output_path.name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to restore parameters from existing result: {e}")
                    selected_peak_angle = None

            # Run alignment (initially clean)
            # Pass selected_peak_angle to find correct index after computing new peaks
            self._run_alignment(
                selected_peak_angle=selected_peak_angle, 
                is_dirty=False
            )

        except InsufficientPinPointsError as e:
            # Switch to preview mode - uncheck auto iteration
            self._param_panel.set_auto_iteration(False)

            # Reload SfM data in preview mode (no iteration)
            sfm_params = self._param_panel.get_sfm_pin_params()
            try:
                self._current_sfm = self._loader.load_sfm(
                    pid,
                    visualize=True,
                    status_callback=self._update_statusbar,
                    initial_thresh=sfm_params.initial_threshold,
                    hsv_weights=sfm_params.hsv_weights,
                    target_hull_volume=sfm_params.target_hull_volume,
                    threshold_callback=self._on_threshold_update,
                    auto_iteration=False,
                )

                # Update Tab 2 visualization only
                self._viewer.set_raw_cloud(
                    self._current_sfm["pcd"], self._current_rgbd["pcd"]
                )
                self._viewer.set_sfm_pin_data(self._current_sfm)

            except Exception as preview_err:
                logger.exception("Failed to load preview")

            # Show warning
            QMessageBox.warning(
                self,
                "Preview Mode",
                f"Initial pin points too few ({e.points_found} points found).\n\n"
                f"Entering preview mode. Adjust 'Initial Threshold' and\n"
                f"observe the red points in Tab 2.\n\n"
                f"Check 'Auto Iteration' to retry with new threshold.",
            )
            self._status_bar.showMessage(
                f"{pid} - Preview mode (adjust threshold)"
            )

        except Exception as e:
            logger.exception(f"Failed to load {pid}")
            QMessageBox.critical(self, "Error", f"Failed to load {pid}:\n{e}")
            self._status_bar.showMessage(f"Error loading {pid}")

    def _run_alignment(
        self, 
        selected_peak_angle: Optional[float] = None, 
        is_dirty: bool = True
    ) -> None:
        """
        Run the alignment pipeline.
        
        Parameters
        ----------
        selected_peak_angle : float, optional
            The rotation angle to select. If None, auto-selects the best peak (lowest RMSE).
            This is used for backward compatibility with old JSON files that store angles.
        is_dirty : bool
            Whether to mark the result as modified (unsaved).
        """
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
            # First compute with None to get all peaks
            self._current_result = self._aligner.compute_full_alignment(
                self._current_rgbd,
                self._current_sfm,
                selected_peak=None,  # Auto-select best initially
                invert_sfm=self._invert_sfm_vector,
                invert_rgbd=self._invert_rgbd_vector,
            )
            
            # Default to auto-selected angle
            curr_idx = self._current_result.selected_peak_idx
            if 0 <= curr_idx < len(self._current_result.peak_angles):
                self._current_selected_angle = self._current_result.peak_angles[curr_idx]
            
            # If a specific angle was requested, find its index and recompute
            if selected_peak_angle is not None:
                peak_angles = self._current_result.peak_angles
                # Find the index of the requested angle in auto-detected peaks
                matching_indices = np.where(np.isclose(peak_angles, selected_peak_angle))[0]
                
                if len(matching_indices) > 0:
                    # Found in auto-detected peaks
                    target_peak_idx = int(matching_indices[0])
                    logger.info(
                        f"Found matching peak at index {target_peak_idx} "
                        f"for angle {selected_peak_angle}°"
                    )
                    self._current_result = self._aligner.recompute_with_peak(
                        target_peak_idx,
                        self._current_rgbd,
                        self._current_sfm,
                    )
                    self._current_selected_angle = selected_peak_angle
                elif selected_peak_angle in self._manual_specified_angles:
                    # Manual angle - compute alignment using direct angle index
                    logger.info(
                        f"Computing alignment for manual angle {selected_peak_angle}°"
                    )
                    self._current_result = self._recompute_with_manual_angle(
                        selected_peak_angle
                    )
                    self._current_selected_angle = selected_peak_angle
                else:
                    logger.warning(
                        f"Could not find peak angle {selected_peak_angle}° in computed peaks. "
                        f"Available: {peak_angles}. Using auto-selected best peak."
                    )
            
            logger.success(f"Alignment complete. RMSE: {self._current_result.rmse}")

            # Update views
            self._viewer.set_transform(self._current_result.transform_matrix)
            
            # Update Pin Detection View (Step 3) with potentially new vectors
            pin_detect_data = {
                # SfM Group
                'sfm_pcd': self._current_sfm['pcd'],
                'sfm_pin_pcd': self._current_sfm['pin_pcd'],
                'sfm_disk': self._current_result.sfm_pin_data.get('circle_mesh'),
                'sfm_arrow': self._current_result.sfm_pin_data.get('vector_arrow'),
                
                # RGBD Group
                'rgbd_pcd': self._current_rgbd['pcd'],
                'rgbd_pin_pcd': self._current_rgbd['pin_pcd'],
                'rgbd_disk': self._current_result.rgbd_pin_data.get('circle_mesh'),
                'rgbd_arrow': self._current_result.rgbd_pin_data.get('vector_arrow'),
            }
            self._viewer.set_pin_detection_data(pin_detect_data)
            
            # Update dirty state
            self._is_dirty = is_dirty
            
            # Update Table
            self._param_panel.set_transform_matrix(
                self._current_result.transform_matrix, 
                modified=self._is_dirty
            )

            # Update chart
            angles, rmses = self._current_result.rmse_curve
            # Get peak indices from peak angles
            peak_indices = []
            for pa in self._current_result.peak_angles:
                idx = int(pa / 10) - 1
                if 0 <= idx < len(angles):
                    peak_indices.append(idx)
            
            # Merge manual specified angles into peaks and sort by angle
            # First, collect manual peak indices
            manual_angle_indices = []
            for ma in self._manual_specified_angles:
                manual_idx = int(ma / 10) - 1
                if 0 <= manual_idx < len(angles) and manual_idx not in peak_indices:
                    peak_indices.append(manual_idx)
                    manual_angle_indices.append(manual_idx)
            
            # Sort peak_indices by corresponding angle values
            peak_indices.sort(key=lambda idx: angles[idx])
            
            # Rebuild manual_potential_indices after sorting
            # Find where manual peaks ended up in the sorted list
            manual_potential_indices = []
            for i, peak_idx in enumerate(peak_indices):
                if peak_idx in manual_angle_indices:
                    manual_potential_indices.append(i)
            
            # Store manual indices in result for saving
            self._current_result.manual_potential_indices = manual_potential_indices
            
            # Build manual_peak_flags for chart (True for manual peaks)
            manual_peak_flags = np.zeros(len(peak_indices), dtype=bool)
            for mi in manual_potential_indices:
                if mi < len(manual_peak_flags):
                    manual_peak_flags[mi] = True
            
            # Determine the correct selected peak index for chart
            chart_selected_idx = self._current_result.selected_peak_idx
            
            # If a specific angle was selected, find its index in the sorted list
            if selected_peak_angle is not None:
                target_idx = int(selected_peak_angle / 10) - 1
                for i, peak_idx in enumerate(peak_indices):
                    if peak_idx == target_idx:
                        chart_selected_idx = i
                        break
                    
            logger.debug(
                f"Updating chart with {len(angles)} points, "
                f"{len(peak_indices)} peaks, {len(manual_potential_indices)} manual, "
                f"selected={chart_selected_idx}"
            )
            
            self._rmse_chart.set_data(
                angles,
                rmses,
                peak_indices,
                chart_selected_idx,
                manual_peak_flags,
            )

            # Retrieve angle for selected peak
            current_idx = self._current_result.selected_peak_idx
            if 0 <= current_idx < len(self._current_result.peak_angles):
                angle = self._current_result.peak_angles[current_idx]
                peak_info = f", (Peak {current_idx}: {angle}°)"
            else:
                peak_info = ""

            self._status_bar.showMessage(
                f"{self._current_pid}: RMSE={self._current_result.rmse:.6f}{peak_info}"
            )
            if self._is_dirty:
                self._status_bar.showMessage(f"{self._current_pid} (Modified)")

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

    def _recompute_with_manual_angle(self, angle: float):
        """
        Compute alignment for a manually specified angle.

        Uses the cached NUV matrices to compute alignment for an angle
        that may not be in the auto-detected peaks list.

        Parameters
        ----------
        angle : float
            The rotation angle in degrees.

        Returns
        -------
        AlignmentResult
            Updated alignment result for the specified angle.
        """
        import copy
        from utils import linear_algebra as util_la
        from core.alignment import AlignmentResult

        last = self._current_result
        angles, rmses = last.rmse_curve

        # Find the index for this angle in the NUV matrices
        angle_idx = int(angle / 10) - 1
        if angle_idx < 0 or angle_idx >= len(last.nuv_matrices):
            logger.warning(f"Invalid angle index {angle_idx} for angle {angle}°")
            return last

        nuv_matrix = last.nuv_matrices[angle_idx]

        # Recompute rough alignment
        imatrix = self._aligner.compute_rough_alignment(
            last.rgbd_pin_data,
            last.sfm_pin_data,
        )
        iimatrix = nuv_matrix @ imatrix

        # Recompute ICP
        tmatrix, o3d_rmse = self._aligner.compute_icp_refinement(
            self._current_rgbd, self._current_sfm, iimatrix
        )

        source_pcd = copy.deepcopy(self._current_rgbd["pcd"]).transform(tmatrix)
        rmse = util_la.compute_distance_rmse(source_pcd, self._current_sfm["pcd"])

        # Find the corresponding selected_peak_idx in the result
        # This is used for display purposes
        selected_peak_idx = last.selected_peak_idx

        return AlignmentResult(
            transform_matrix=tmatrix,
            rmse=rmse,
            open3d_rmse=o3d_rmse,
            peak_angles=last.peak_angles,
            rmse_curve=last.rmse_curve,
            selected_peak_idx=selected_peak_idx,
            sfm_pin_data=last.sfm_pin_data,
            rgbd_pin_data=last.rgbd_pin_data,
            nuv_matrices=last.nuv_matrices,
            manual_potential_indices=last.manual_potential_indices,
        )

    def _on_threshold_update(
        self, threshold: float, dbscan_activated: bool = False
    ) -> None:
        """
        Handle threshold updates during iterative pin segmentation.

        Updates the 'Current Threshold' label in the parameter panel
        in real-time as the algorithm adjusts the threshold.

        Parameters
        ----------
        threshold : float
            Current threshold value being used.
        dbscan_activated : bool
            Whether DBSCAN clustering was activated.
        """
        self._param_panel.set_current_threshold(threshold, dbscan_activated)
        QApplication.processEvents()

    @Slot(object)
    def _on_params_changed(self, params: AlignmentParams) -> None:
        """Handle parameter changes for real-time update."""
        if self._current_result is None:
            return

        # Recompute ICP with new parameters, preserving current peak selection
        if self._aligner is not None:
            self._aligner.update_params(params)
            # Preserve current selected peak angle
            current_angle = self._current_selected_angle
            self._run_alignment(selected_peak_angle=current_angle, is_dirty=True)

    @Slot(object)
    def _on_sfm_pin_params_changed(self, sfm_params) -> None:
        """
        Handle Step 2 SfM pin parameter changes.

        Reloads the SfM data with new HSV parameters when the user modifies
        controls in the Step 2 panel.

        Parameters
        ----------
        sfm_params : SfMPinParams
            New SfM pin segmentation parameters.
        """
        if self._loader is None or self._current_pid is None:
            return

        self._status_bar.showMessage(
            f"Reloading SfM data for {self._current_pid}..."
        )
        QApplication.processEvents()

        try:
            # Reload SfM data with new parameters
            self._current_sfm = self._loader.load_sfm(
                self._current_pid,
                visualize=True,
                status_callback=self._update_statusbar,
                initial_thresh=sfm_params.initial_threshold,
                hsv_weights=sfm_params.hsv_weights,
                target_hull_volume=sfm_params.target_hull_volume,
                threshold_callback=self._on_threshold_update,
                auto_iteration=sfm_params.auto_iteration,
            )

            logger.info("Reloaded SfM data: {}", self._current_sfm)

            # Update viewer with new SfM data
            self._viewer.set_raw_cloud(
                self._current_sfm["pcd"], self._current_rgbd["pcd"]
            )
            self._viewer.set_sfm_pin_data(self._current_sfm)

            # Re-run alignment only if auto_iteration is enabled
            if sfm_params.auto_iteration:
                self._run_alignment(selected_peak_angle=None, is_dirty=True)
            else:
                # Preview mode - just show visualization
                self._status_bar.showMessage(
                    f"{self._current_pid} - Preview mode (adjust threshold)"
                )

        except InsufficientPinPointsError as e:
            # Switch to preview mode
            self._param_panel.set_auto_iteration(False)

            # Reload in preview mode
            try:
                self._current_sfm = self._loader.load_sfm(
                    self._current_pid,
                    visualize=True,
                    status_callback=self._update_statusbar,
                    initial_thresh=sfm_params.initial_threshold,
                    hsv_weights=sfm_params.hsv_weights,
                    target_hull_volume=sfm_params.target_hull_volume,
                    threshold_callback=self._on_threshold_update,
                    auto_iteration=False,
                )
                self._viewer.set_raw_cloud(
                    self._current_sfm["pcd"], self._current_rgbd["pcd"]
                )
                self._viewer.set_sfm_pin_data(self._current_sfm)
            except Exception:
                logger.exception("Failed to load preview")

            QMessageBox.warning(
                self,
                "Preview Mode",
                f"Pin points too few ({e.points_found} points found).\n\n"
                f"Entering preview mode. Adjust threshold.",
            )
            self._status_bar.showMessage(
                f"{self._current_pid} - Preview mode (adjust threshold)"
            )

        except Exception as e:
            logger.exception(f"Failed to reload SfM data")
            QMessageBox.critical(
                self, "Error", f"Failed to reload SfM data:\n{e}"
            )
            self._status_bar.showMessage("Reload failed")

    @Slot(int)
    def _on_peak_changed(self, peak_idx: int) -> None:
        """Handle peak selection change from chart navigation."""
        if self._aligner is None or self._current_rgbd is None:
            return

        # Get merged peak list (auto + manual) from chart
        chart_peaks = self._rmse_chart._peaks
        angles, _ = self._current_result.rmse_curve
        
        # Determine if this is a manual peak or auto peak
        if 0 <= peak_idx < len(chart_peaks):
            angle_idx = chart_peaks[peak_idx]
            selected_angle = int(angles[angle_idx])
            
            # Update the tracked selected angle
            self._current_selected_angle = selected_angle
            
            # Check if it's a manual angle
            if selected_angle in self._manual_specified_angles:
                # Manual angle - recompute using manual angle logic
                self._current_result = self._recompute_with_manual_angle(
                    float(selected_angle)
                )
            else:
                # Auto-detected peak - use standard recompute
                # Find the index in auto-detected peaks
                peak_angles = self._current_result.peak_angles
                matching = np.where(np.isclose(peak_angles, selected_angle))[0]
                if len(matching) > 0:
                    auto_peak_idx = int(matching[0])
                    self._current_result = self._aligner.recompute_with_peak(
                        auto_peak_idx,
                        self._current_rgbd,
                        self._current_sfm,
                    )
                else:
                    # Fallback: use the angle index directly
                    self._current_result = self._recompute_with_manual_angle(
                        float(selected_angle)
                    )
            
            peak_info = f", (Peak {peak_idx}: {selected_angle}°)"
        else:
            selected_angle = None
            peak_info = ""

        self._viewer.set_transform(self._current_result.transform_matrix)
        self._status_bar.showMessage(
            f"{self._current_pid}: RMSE={self._current_result.rmse:.6f}{peak_info}"
        )
        self._is_dirty = True
        self._param_panel.set_transform_matrix(
            self._current_result.transform_matrix, 
            modified=True
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
                peak_angles=self._current_result.peak_angles,
                selected_peak_idx=self._current_result.selected_peak_idx,
                manual_potential_indices=self._current_result.manual_potential_indices,
            )

            self._file_tree.set_completed(self._current_pid, True)
            self._status_bar.showMessage(f"Saved {self._current_pid}")
            
            # Reset dirty state
            self._is_dirty = False
            self._param_panel.set_transform_matrix(
                self._current_result.transform_matrix, 
                modified=False
            )
            
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
        dialog.set_point_size_modifier(
            self._settings.value(
                "point_size_modifier",
                PreferencesDialog.DEFAULT_MODIFIER,
            )
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

            # Save and apply point size modifier
            modifier_name = dialog.get_point_size_modifier_name()
            self._settings.setValue("point_size_modifier", modifier_name)
            self._apply_point_size_modifier(modifier_name)

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

    def _apply_point_size_modifier(self, modifier_name: str) -> None:
        """
        Apply the point size modifier key to the 3D viewer.

        Parameters
        ----------
        modifier_name : str
            Modifier key name ("Alt", "Ctrl", or "Shift").
        """
        logger.debug(f"[PointSize] _apply_point_size_modifier called with: {modifier_name}")
        modifier = PreferencesDialog.MODIFIER_OPTIONS.get(
            modifier_name,
            PreferencesDialog.MODIFIER_OPTIONS[PreferencesDialog.DEFAULT_MODIFIER],
        )
        logger.debug(f"[PointSize] Resolved modifier value: {modifier.value} (Qt.AltModifier={Qt.AltModifier.value})")
        self._viewer.set_point_size_modifier(modifier)
        logger.info(f"[PointSize] Point size modifier set to: {modifier_name} (value={modifier.value})")

    @Slot()
    def _on_reset_parameters(self) -> None:
        """Reset all parameters in the Parameters panel to their default values."""
        self._param_panel.reset_to_defaults()
        self._status_bar.showMessage("Parameters reset to defaults.")
        logger.info("Parameters reset to default values.")

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

    @Slot()
    def _on_manual_angle_specify(self) -> None:
        """
        Open dialog to manually specify a rotation angle.

        Adds the specified angle to the potential local minima list
        and automatically selects it.
        """
        if self._current_result is None:
            QMessageBox.warning(
                self,
                "Warning",
                "Please load an item first before adding manual angles.",
            )
            return

        # Get existing angles (auto-detected + manual)
        existing_angles = set(self._current_result.peak_angles)
        existing_angles.update(self._manual_specified_angles)

        # Create list of available angle options (exclude existing)
        all_angles = list(range(0, 360, 10))
        available_angles = [a for a in all_angles if a not in existing_angles]

        if not available_angles:
            QMessageBox.information(
                self,
                "Information",
                "All angles are already in use.",
            )
            return

        angle_options = [str(a) for a in available_angles]

        angle_str, ok = QInputDialog.getItem(
            self,
            "Manual Specify Rotation Angle",
            "Select rotation angle (0-350°, step 10°):",
            angle_options,
            current=0,
            editable=False,
        )

        if not ok:
            return

        angle = int(angle_str)

        # Add to manual angles
        self._manual_specified_angles.append(angle)
        logger.info(f"Added manual angle: {angle}°")

        # Re-run alignment with the new angle selected
        # The angle will be inserted in sorted order and selected correctly
        self._run_alignment(
            selected_peak_angle=float(angle),
            is_dirty=True,
        )
        self._status_bar.showMessage(
            f"Added and selected manual angle {angle}°."
        )

    @Slot()
    def _on_reset_manual_angles(self) -> None:
        """
        Clear all manually specified angles.

        Restores the chart to show only auto-detected local minima.
        """
        if not self._manual_specified_angles:
            self._status_bar.showMessage("No manual angles to reset.")
            return

        count = len(self._manual_specified_angles)
        self._manual_specified_angles = []
        logger.info(f"Reset {count} manual angle(s)")

        # Re-run alignment to update chart
        if self._current_result is not None:
            self._run_alignment(
                selected_peak_angle=None,  # Auto-select best peak
                is_dirty=True,
            )

        self._status_bar.showMessage(f"Reset {count} manual angle(s).")

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        self._save_settings()
        self._viewer.close_plotters()
        super().closeEvent(event)
