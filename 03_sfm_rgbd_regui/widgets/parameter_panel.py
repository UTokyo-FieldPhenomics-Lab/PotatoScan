# -*- coding: utf-8 -*-
"""
Parameter panel widget for alignment settings.

Provides real-time parameter adjustment with signal emission.
"""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal, Slot, QTimer, Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)
import numpy as np

# Add parent for core imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.alignment import AlignmentParams, SfMPinParams


class ParameterPanel(QWidget):
    """
    Parameter adjustment panel with real-time signal updates.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.

    Signals
    -------
    params_changed : Signal(AlignmentParams)
        Emitted when any parameter changes (debounced).

    Attributes
    ----------
    DEFAULT_PARAMS : AlignmentParams
        Default parameter values from AlignmentParams dataclass.

    Examples
    --------
    >>> panel = ParameterPanel()
    >>> panel.params_changed.connect(on_params_change)
    >>> params = panel.get_params()
    """

    # Default values from AlignmentParams dataclass
    DEFAULT_PARAMS: AlignmentParams = AlignmentParams()

    # Default values for SfM pin segmentation
    DEFAULT_SFM_PIN_PARAMS: SfMPinParams = SfMPinParams()

    params_changed = Signal(object)  # AlignmentParams
    sfm_pin_params_changed = Signal(object)  # SfMPinParams

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the parameter panel."""
        super().__init__(parent)
        self._updating = False

        # Debounce timer for Step 3/4 params
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(800)  # 800ms delay
        self._update_timer.timeout.connect(self._emit_params_now)

        # Debounce timer for Step 2 SfM pin params
        self._sfm_update_timer = QTimer(self)
        self._sfm_update_timer.setSingleShot(True)
        self._sfm_update_timer.setInterval(800)  # 800ms delay
        self._sfm_update_timer.timeout.connect(self._emit_sfm_pin_params_now)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Step 2: SfM Pin Segmentation Group
        sfm_group = QGroupBox("Step 2: SfM Pin Segmentation")
        sfm_layout = QFormLayout(sfm_group)

        # Initial Threshold
        self._spin_initial_thresh = QDoubleSpinBox()
        self._spin_initial_thresh.setRange(0.05, 0.80)
        self._spin_initial_thresh.setSingleStep(0.05)
        self._spin_initial_thresh.setDecimals(2)
        self._spin_initial_thresh.setValue(0.35)
        sfm_layout.addRow("Initial Threshold:", self._spin_initial_thresh)

        # Current Threshold (read-only label)
        self._lbl_current_thresh = QLabel("0.35")
        self._lbl_current_thresh.setStyleSheet("font-weight: bold; color: #666;")
        sfm_layout.addRow("Current Threshold:", self._lbl_current_thresh)

        # HSV Weights - using HBoxLayout for compact display
        hsv_widget = QWidget()
        hsv_layout = QHBoxLayout(hsv_widget)
        hsv_layout.setContentsMargins(0, 0, 0, 0)
        hsv_layout.setSpacing(4)

        # H weight
        self._spin_hsv_h = QDoubleSpinBox()
        self._spin_hsv_h.setRange(0.0, 1.0)
        self._spin_hsv_h.setSingleStep(0.1)
        self._spin_hsv_h.setDecimals(1)
        self._spin_hsv_h.setValue(0.8)
        self._spin_hsv_h.setPrefix("H:")
        hsv_layout.addWidget(self._spin_hsv_h)

        # S weight
        self._spin_hsv_s = QDoubleSpinBox()
        self._spin_hsv_s.setRange(0.0, 1.0)
        self._spin_hsv_s.setSingleStep(0.1)
        self._spin_hsv_s.setDecimals(1)
        self._spin_hsv_s.setValue(0.1)
        self._spin_hsv_s.setPrefix("S:")
        hsv_layout.addWidget(self._spin_hsv_s)

        # V weight
        self._spin_hsv_v = QDoubleSpinBox()
        self._spin_hsv_v.setRange(0.0, 1.0)
        self._spin_hsv_v.setSingleStep(0.1)
        self._spin_hsv_v.setDecimals(1)
        self._spin_hsv_v.setValue(0.1)
        self._spin_hsv_v.setPrefix("V:")
        hsv_layout.addWidget(self._spin_hsv_v)

        sfm_layout.addRow("HSV Weights:", hsv_widget)

        # Target Hull Volume
        self._spin_hull_volume = QDoubleSpinBox()
        self._spin_hull_volume.setRange(10.0, 500.0)
        self._spin_hull_volume.setSingleStep(10.0)
        self._spin_hull_volume.setDecimals(1)
        self._spin_hull_volume.setValue(100.0)
        self._spin_hull_volume.setSuffix(" mm³")
        sfm_layout.addRow("Target Hull Volume:", self._spin_hull_volume)

        # Auto Iteration checkbox
        self._chk_auto_iter = QCheckBox("Enable iterative threshold reduction")
        self._chk_auto_iter.setChecked(True)
        sfm_layout.addRow("Auto Iteration:", self._chk_auto_iter)

        layout.addWidget(sfm_group)

        # Step 3: Pin Neighbor Group
        pin_group = QGroupBox("Step 3: Pin Neighbor")
        pin_layout = QFormLayout(pin_group)

        self._spin_search_radius = QDoubleSpinBox()
        self._spin_search_radius.setRange(0.001, 0.10)
        self._spin_search_radius.setSingleStep(0.001)
        self._spin_search_radius.setDecimals(4)
        self._spin_search_radius.setValue(0.01)
        self._spin_search_radius.setSuffix(" m")
        pin_layout.addRow("Search Radius:", self._spin_search_radius)

        self._spin_cross_buffer = QDoubleSpinBox()
        self._spin_cross_buffer.setRange(0.0005, 0.005)
        self._spin_cross_buffer.setSingleStep(0.0005)
        self._spin_cross_buffer.setDecimals(4)
        self._spin_cross_buffer.setValue(0.001)
        self._spin_cross_buffer.setSuffix(" m")
        pin_layout.addRow("Cross Buffer:", self._spin_cross_buffer)

        layout.addWidget(pin_group)

        # ICP Group
        icp_group = QGroupBox("Step 4: Colored-ICP Refinement")
        icp_layout = QFormLayout(icp_group)

        self._spin_icp_threshold = QDoubleSpinBox()
        self._spin_icp_threshold.setRange(0.0005, 0.005)
        self._spin_icp_threshold.setSingleStep(0.0005)
        self._spin_icp_threshold.setDecimals(4)
        self._spin_icp_threshold.setValue(0.001)
        self._spin_icp_threshold.setSuffix(" m")
        icp_layout.addRow("Threshold:", self._spin_icp_threshold)

        self._spin_icp_iter = QSpinBox()
        self._spin_icp_iter.setRange(0, 100)
        self._spin_icp_iter.setSingleStep(1)
        self._spin_icp_iter.setValue(0)
        icp_layout.addRow("Iterations:", self._spin_icp_iter)

        # Geometry weight slider
        weight_widget = QWidget()
        weight_layout = QVBoxLayout(weight_widget)
        weight_layout.setContentsMargins(0, 0, 0, 0)

        self._slider_weight = QSlider(Qt.Horizontal)
        self._slider_weight.setRange(0, 100)
        self._slider_weight.setValue(10)
        self._slider_weight.setTickPosition(QSlider.TicksBelow)
        self._slider_weight.setTickInterval(10)

        self._label_weight = QLabel("0.10")
        self._label_weight.setAlignment(Qt.AlignCenter)

        weight_layout.addWidget(self._slider_weight)
        weight_layout.addWidget(self._label_weight)

        icp_layout.addRow("Geometry Weight:", weight_widget)

        layout.addWidget(icp_group)
        
        # Transform Matrix Table
        matrix_group = QGroupBox("Transform Matrix")
        matrix_layout = QVBoxLayout(matrix_group)
        matrix_layout.setAlignment(Qt.AlignCenter)

        self._table_matrix = QTableWidget(4, 4)
        self._table_matrix.verticalHeader().setVisible(False)
        self._table_matrix.horizontalHeader().setVisible(False)
        self._table_matrix.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table_matrix.setSelectionMode(QTableWidget.NoSelection)
        self._table_matrix.setFocusPolicy(Qt.NoFocus)

        # Disable scrollbars to prevent whitespace issues
        self._table_matrix.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._table_matrix.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Stretch rows and columns to fill the fixed size
        self._table_matrix.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table_matrix.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Fixed size (width fits in 300px panel, height fits 4 rows)
        self._table_matrix.setFixedSize(260, 125)

        # Initialize identity
        self.set_transform_matrix(np.eye(4), modified=False)
        matrix_layout.addWidget(self._table_matrix)
        
        layout.addWidget(matrix_group)
        layout.addStretch()

    def set_transform_matrix(self, matrix: np.ndarray, modified: bool = True) -> None:
        """
        Update the transform matrix table.

        Parameters
        ----------
        matrix : np.ndarray
            4x4 transformation matrix.
        modified : bool, optional
            If True, text color is green (unsaved/calc). If False, black (saved/default).
        """
        color = QColor("green") if modified else QColor("black")
        brush = QBrush(color)

        for i in range(4):
            for j in range(4):
                val = matrix[i, j]
                item = self._table_matrix.item(i, j)
                if not item:
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignCenter)
                    self._table_matrix.setItem(i, j, item)
                
                item.setText(f"{val:.4f}")
                item.setForeground(brush)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        # Step 2: SfM Pin params - value changes trigger timer (debounce)
        self._spin_initial_thresh.valueChanged.connect(self._on_sfm_param_changed)
        self._spin_hsv_h.valueChanged.connect(self._on_sfm_param_changed)
        self._spin_hsv_s.valueChanged.connect(self._on_sfm_param_changed)
        self._spin_hsv_v.valueChanged.connect(self._on_sfm_param_changed)
        self._spin_hull_volume.valueChanged.connect(self._on_sfm_param_changed)

        # Step 2: Explicit triggers (Enter key or Focus lost) - Immediate update
        self._spin_initial_thresh.editingFinished.connect(
            self._emit_sfm_pin_params_now
        )
        self._spin_hsv_h.editingFinished.connect(self._emit_sfm_pin_params_now)
        self._spin_hsv_s.editingFinished.connect(self._emit_sfm_pin_params_now)
        self._spin_hsv_v.editingFinished.connect(self._emit_sfm_pin_params_now)
        self._spin_hull_volume.editingFinished.connect(
            self._emit_sfm_pin_params_now
        )

        # Auto Iteration checkbox - immediate emit on toggle
        self._chk_auto_iter.stateChanged.connect(self._emit_sfm_pin_params_now)

        # Step 3/4: Value changes trigger timer (debounce)
        self._spin_search_radius.valueChanged.connect(self._on_param_changed)
        self._spin_cross_buffer.valueChanged.connect(self._on_param_changed)
        self._spin_icp_threshold.valueChanged.connect(self._on_param_changed)
        self._spin_icp_iter.valueChanged.connect(self._on_param_changed)
        self._slider_weight.valueChanged.connect(self._on_weight_changed)

        # Step 3/4: Explicit triggers (Enter key or Focus lost) - Immediate
        self._spin_search_radius.editingFinished.connect(self._emit_params_now)
        self._spin_cross_buffer.editingFinished.connect(self._emit_params_now)
        self._spin_icp_threshold.editingFinished.connect(self._emit_params_now)
        self._spin_icp_iter.editingFinished.connect(self._emit_params_now)
        self._slider_weight.sliderReleased.connect(self._emit_params_now)

    @Slot()
    def _on_sfm_param_changed(self) -> None:
        """Handle Step 2 SfM pin parameter changes (debounced)."""
        if not self._updating:
            self._sfm_update_timer.start()

    @Slot()
    def _on_param_changed(self) -> None:
        """Handle Step 3/4 parameter value changes (debounced)."""
        if not self._updating:
            self._update_timer.start()

    @Slot(int)
    def _on_weight_changed(self, value: int) -> None:
        """Handle weight slider changes (debounced)."""
        weight = value / 100.0
        self._label_weight.setText(f"{weight:.2f}")
        if not self._updating:
            self._update_timer.start()

    @Slot()
    def _emit_sfm_pin_params_now(self) -> None:
        """Emit current SfM pin parameters immediately."""
        if self._sfm_update_timer.isActive():
            self._sfm_update_timer.stop()
        self.sfm_pin_params_changed.emit(self.get_sfm_pin_params())

    @Slot()
    def _emit_params_now(self) -> None:
        """Emit current parameters immediately, stopping pending timer."""
        # Avoid redundant emits if timer was just about to fire or already fired
        if self._update_timer.isActive():
            self._update_timer.stop()
        
        # Emit signal
        self.params_changed.emit(self.get_params())

    def get_params(self) -> AlignmentParams:
        """
        Get current parameter values.

        Returns
        -------
        AlignmentParams
            Current alignment parameters.
        """
        return AlignmentParams(
            search_radius=self._spin_search_radius.value(),
            cross_buffer=self._spin_cross_buffer.value(),
            icp_threshold=self._spin_icp_threshold.value(),
            icp_iter_num=self._spin_icp_iter.value(),
            geometry_weight=self._slider_weight.value() / 100.0,
        )

    def set_params(self, params: AlignmentParams) -> None:
        """
        Set parameter values.

        Parameters
        ----------
        params : AlignmentParams
            Parameters to set.
        """
        self._updating = True
        try:
            self._spin_search_radius.setValue(params.search_radius)
            self._spin_cross_buffer.setValue(params.cross_buffer)
            self._spin_icp_threshold.setValue(params.icp_threshold)
            self._spin_icp_iter.setValue(params.icp_iter_num)
            self._slider_weight.setValue(int(params.geometry_weight * 100))
            self._label_weight.setText(f"{params.geometry_weight:.2f}")
        finally:
            self._updating = False

    def reset_to_defaults(self) -> None:
        """
        Reset all parameters to their default values.

        Resets all parameter input controls to the values defined in
        `AlignmentParams` and `SfMPinParams` dataclass defaults.
        Emits signals to notify listeners of the change.

        Examples
        --------
        >>> panel = ParameterPanel()
        >>> panel.reset_to_defaults()
        """
        self.set_params(self.DEFAULT_PARAMS)
        self.set_sfm_pin_params(self.DEFAULT_SFM_PIN_PARAMS)
        self.params_changed.emit(self.get_params())
        self.sfm_pin_params_changed.emit(self.get_sfm_pin_params())

    def get_sfm_pin_params(self) -> SfMPinParams:
        """
        Get current SfM pin segmentation parameter values.

        Returns
        -------
        SfMPinParams
            Current SfM pin parameters.
        """
        return SfMPinParams(
            initial_threshold=self._spin_initial_thresh.value(),
            hsv_weight_h=self._spin_hsv_h.value(),
            hsv_weight_s=self._spin_hsv_s.value(),
            hsv_weight_v=self._spin_hsv_v.value(),
            target_hull_volume=self._spin_hull_volume.value(),
            auto_iteration=self._chk_auto_iter.isChecked(),
        )

    def set_sfm_pin_params(self, params: SfMPinParams) -> None:
        """
        Set SfM pin segmentation parameter values.

        Parameters
        ----------
        params : SfMPinParams
            SfM pin parameters to set.
        """
        self._updating = True
        try:
            self._spin_initial_thresh.setValue(params.initial_threshold)
            self._spin_hsv_h.setValue(params.hsv_weight_h)
            self._spin_hsv_s.setValue(params.hsv_weight_s)
            self._spin_hsv_v.setValue(params.hsv_weight_v)
            self._spin_hull_volume.setValue(params.target_hull_volume)
            self._chk_auto_iter.setChecked(params.auto_iteration)
            self._lbl_current_thresh.setText(f"{params.initial_threshold:.2f}")
        finally:
            self._updating = False

    def set_current_threshold(
        self, value: float, dbscan_activated: bool = False
    ) -> None:
        """
        Update the current threshold display label.

        Called during iterative pin segmentation to show the dynamically
        adjusted threshold value.

        Parameters
        ----------
        value : float
            Current threshold value being used.
        dbscan_activated : bool
            Whether DBSCAN clustering was activated.
        """
        suffix = " (DBSCAN activated)" if dbscan_activated else ""
        self._lbl_current_thresh.setText(f"{value:.2f}{suffix}")

    def set_auto_iteration(self, enabled: bool) -> None:
        """
        Set the auto iteration checkbox state without emitting signals.

        Parameters
        ----------
        enabled : bool
            Whether auto iteration should be enabled.
        """
        self._updating = True
        try:
            self._chk_auto_iter.setChecked(enabled)
        finally:
            self._updating = False

    def reset_step2_and_step3(self) -> None:
        """
        Reset only Step 2 (SfM Pin) and Step 3 (Pin Neighbor) parameters.

        Preserves Step 4 (ICP) parameters. This is called when loading a
        new item to ensure fresh segmentation while keeping ICP settings.
        """
        self._updating = True
        try:
            # Reset Step 2: SfM Pin Segmentation
            defaults_sfm = self.DEFAULT_SFM_PIN_PARAMS
            self._spin_initial_thresh.setValue(defaults_sfm.initial_threshold)
            self._spin_hsv_h.setValue(defaults_sfm.hsv_weight_h)
            self._spin_hsv_s.setValue(defaults_sfm.hsv_weight_s)
            self._spin_hsv_v.setValue(defaults_sfm.hsv_weight_v)
            self._spin_hull_volume.setValue(defaults_sfm.target_hull_volume)
            self._lbl_current_thresh.setText(
                f"{defaults_sfm.initial_threshold:.2f}"
            )

            # Reset Step 3: Pin Neighbor
            defaults_align = self.DEFAULT_PARAMS
            self._spin_search_radius.setValue(defaults_align.search_radius)
            self._spin_cross_buffer.setValue(defaults_align.cross_buffer)
        finally:
            self._updating = False

    def reset_all_steps(self) -> None:
        """
        Reset Step 2, Step 3, and Step 4 parameters to defaults.

        Called when loading a new item to ensure completely fresh state.
        """
        self._updating = True
        try:
            # Reset Step 2: SfM Pin Segmentation
            defaults_sfm = self.DEFAULT_SFM_PIN_PARAMS
            self._spin_initial_thresh.setValue(defaults_sfm.initial_threshold)
            self._spin_hsv_h.setValue(defaults_sfm.hsv_weight_h)
            self._spin_hsv_s.setValue(defaults_sfm.hsv_weight_s)
            self._spin_hsv_v.setValue(defaults_sfm.hsv_weight_v)
            self._spin_hull_volume.setValue(defaults_sfm.target_hull_volume)
            self._chk_auto_iter.setChecked(defaults_sfm.auto_iteration)
            self._lbl_current_thresh.setText(
                f"{defaults_sfm.initial_threshold:.2f}"
            )

            # Reset Step 3: Pin Neighbor
            defaults_align = self.DEFAULT_PARAMS
            self._spin_search_radius.setValue(defaults_align.search_radius)
            self._spin_cross_buffer.setValue(defaults_align.cross_buffer)

            # Reset Step 4: ICP Refinement
            self._spin_icp_threshold.setValue(defaults_align.icp_threshold)
            self._spin_icp_iter.setValue(defaults_align.icp_iter_num)
            self._slider_weight.setValue(int(defaults_align.geometry_weight * 100))
            self._label_weight.setText(f"{defaults_align.geometry_weight:.2f}")
        finally:
            self._updating = False
