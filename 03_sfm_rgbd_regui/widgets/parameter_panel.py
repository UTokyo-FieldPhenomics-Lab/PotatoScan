# -*- coding: utf-8 -*-
"""
Parameter panel widget for alignment settings.

Provides real-time parameter adjustment with signal emission.
"""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

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
        Emitted when any parameter changes.

    Examples
    --------
    >>> panel = ParameterPanel()
    >>> panel.params_changed.connect(on_params_change)
    >>> params = panel.get_params()
    """

    params_changed = Signal(object)  # AlignmentParams

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the parameter panel."""
        super().__init__(parent)
        self._updating = False
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
        self._spin_search_radius.setRange(0.01, 0.10)
        self._spin_search_radius.setSingleStep(0.005)
        self._spin_search_radius.setDecimals(3)
        self._spin_search_radius.setValue(0.03)
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
        layout.addStretch()

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._spin_search_radius.valueChanged.connect(self._on_param_changed)
        self._spin_cross_buffer.valueChanged.connect(self._on_param_changed)
        self._spin_icp_threshold.valueChanged.connect(self._on_param_changed)
        self._spin_icp_iter.valueChanged.connect(self._on_param_changed)
        self._slider_weight.valueChanged.connect(self._on_weight_changed)

    @Slot()
    def _on_param_changed(self) -> None:
        """Handle parameter value changes."""
        if not self._updating:
            self.params_changed.emit(self.get_params())

    @Slot(int)
    def _on_weight_changed(self, value: int) -> None:
        """Handle weight slider changes."""
        weight = value / 100.0
        self._label_weight.setText(f"{weight:.2f}")
        if not self._updating:
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
