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
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
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
from core.alignment import AlignmentParams


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

    params_changed = Signal(object)  # AlignmentParams

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the parameter panel."""
        super().__init__(parent)
        self._updating = False
        
        # Debounce timer for updates
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(800)  # 800ms delay
        self._update_timer.timeout.connect(self._emit_params_now)
        
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Pin Neighbor Group
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
        # Value changes trigger timer (debounce)
        self._spin_search_radius.valueChanged.connect(self._on_param_changed)
        self._spin_cross_buffer.valueChanged.connect(self._on_param_changed)
        self._spin_icp_threshold.valueChanged.connect(self._on_param_changed)
        self._spin_icp_iter.valueChanged.connect(self._on_param_changed)
        self._slider_weight.valueChanged.connect(self._on_weight_changed)
        
        # Explicit triggers (Enter key or Focus lost, Slider release) - Immediate update
        self._spin_search_radius.editingFinished.connect(self._emit_params_now)
        self._spin_cross_buffer.editingFinished.connect(self._emit_params_now)
        self._spin_icp_threshold.editingFinished.connect(self._emit_params_now)
        self._spin_icp_iter.editingFinished.connect(self._emit_params_now)
        self._slider_weight.sliderReleased.connect(self._emit_params_now)

    @Slot()
    def _on_param_changed(self) -> None:
        """Handle parameter value changes (debounced)."""
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
        `AlignmentParams` dataclass defaults. Emits `params_changed` signal
        to notify listeners of the change.

        Examples
        --------
        >>> panel = ParameterPanel()
        >>> panel.reset_to_defaults()
        """
        self.set_params(self.DEFAULT_PARAMS)
        self.params_changed.emit(self.get_params())
