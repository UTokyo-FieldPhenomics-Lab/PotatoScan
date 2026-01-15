# -*- coding: utf-8 -*-
"""
RMSE chart widget with peak selection controls.

Displays the RMSE curve with local minima markers and navigation buttons.
"""

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class RmseChartWidget(QWidget):
    """
    RMSE chart with peak selection and navigation.

    Displays line chart with local minima as vertical lines.
    Current selection shown in red, others in gray.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.

    Signals
    -------
    peak_changed : Signal(int)
        Emitted when peak selection changes. Carries peak index.

    Examples
    --------
    >>> chart = RmseChartWidget()
    >>> chart.set_data(angles, rmses, peak_indices)
    >>> chart.peak_changed.connect(on_peak_select)
    """

    peak_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the chart widget."""
        super().__init__(parent)
        self._angles: np.ndarray = np.array([])
        self._rmses: np.ndarray = np.array([])
        self._peaks: np.ndarray = np.array([])
        self._current_peak: int = 0
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Matplotlib figure
        self._figure = Figure(figsize=(6, 2.5), dpi=100)
        self._figure.set_tight_layout(True)
        self._canvas = FigureCanvas(self._figure)
        self._ax = self._figure.add_subplot(111)

        layout.addWidget(self._canvas)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(8, 4, 8, 4)

        self._btn_prev = QPushButton("◀ Prev Peak")
        self._btn_prev.clicked.connect(self._on_prev_peak)
        nav_layout.addWidget(self._btn_prev)

        nav_layout.addStretch()

        self._btn_next = QPushButton("Next Peak ▶")
        self._btn_next.clicked.connect(self._on_next_peak)
        nav_layout.addWidget(self._btn_next)

        layout.addLayout(nav_layout)

    def set_data(
        self,
        angles: np.ndarray,
        rmses: np.ndarray,
        peaks: np.ndarray,
        selected: int = 0,
    ) -> None:
        """
        Set chart data.

        Parameters
        ----------
        angles : np.ndarray
            Array of rotation angles (x-axis).
        rmses : np.ndarray
            Array of RMSE values (y-axis).
        peaks : np.ndarray
            Indices of local minima in angles array.
        selected : int
            Currently selected peak index.
        """
        self._angles = angles
        self._rmses = rmses
        self._peaks = peaks
        self._current_peak = selected
        self._update_chart()
        self._update_buttons()

    def _update_chart(self) -> None:
        """Redraw the chart."""
        self._ax.clear()

        if len(self._angles) == 0:
            self._canvas.draw()
            return

        # Plot RMSE curve
        self._ax.plot(self._angles, self._rmses, "b-", linewidth=1.5, label="RMSE")

        # Plot peak markers
        for i, peak_idx in enumerate(self._peaks):
            angle = self._angles[peak_idx]
            if i == self._current_peak:
                self._ax.axvline(
                    x=angle,
                    color="red",
                    linewidth=2,
                    label="Current",
                )
            else:
                self._ax.axvline(
                    x=angle,
                    color="gray",
                    linewidth=1,
                    alpha=0.6,
                )

        # Labels and title
        self._ax.set_xlabel("Rotation Angle (°)")
        self._ax.set_ylabel("RMSE")
        self._ax.set_title("Rotation Optimization - Local Minima")

        # Legend (only show if we have peaks)
        if len(self._peaks) > 0:
            self._ax.legend(loc="upper right", fontsize=8)

        self._canvas.draw()

    def _update_buttons(self) -> None:
        """Update button enabled states."""
        has_peaks = len(self._peaks) > 1
        self._btn_prev.setEnabled(has_peaks and self._current_peak > 0)
        self._btn_next.setEnabled(
            has_peaks and self._current_peak < len(self._peaks) - 1
        )

    @Slot()
    def _on_prev_peak(self) -> None:
        """Navigate to previous peak."""
        if self._current_peak > 0:
            self._current_peak -= 1
            self._update_chart()
            self._update_buttons()
            self.peak_changed.emit(self._current_peak)

    @Slot()
    def _on_next_peak(self) -> None:
        """Navigate to next peak."""
        if self._current_peak < len(self._peaks) - 1:
            self._current_peak += 1
            self._update_chart()
            self._update_buttons()
            self.peak_changed.emit(self._current_peak)

    def get_selected_peak(self) -> int:
        """
        Get currently selected peak index.

        Returns
        -------
        int
            Current peak index.
        """
        return self._current_peak

    def clear(self) -> None:
        """Clear the chart."""
        self._angles = np.array([])
        self._rmses = np.array([])
        self._peaks = np.array([])
        self._current_peak = 0
        self._ax.clear()
        self._canvas.draw()
        self._update_buttons()
