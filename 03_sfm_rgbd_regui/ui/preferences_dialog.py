# -*- coding: utf-8 -*-
"""
Preferences dialog with shortcut editing and developer options.

Allows users to configure keyboard shortcuts and enable debug logging.
"""

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QKeySequenceEdit,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class PreferencesDialog(QDialog):
    """
    Preferences dialog for shortcut configuration and developer options.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.

    Signals
    -------
    shortcuts_changed : Signal(dict)
        Emitted when shortcuts are changed. Carries action->shortcut dict.

    Examples
    --------
    >>> dialog = PreferencesDialog(parent)
    >>> if dialog.exec() == QDialog.Accepted:
    ...     shortcuts = dialog.get_shortcuts()
    ...     dev_mode = dialog.get_developer_mode()
    """

    shortcuts_changed = Signal(dict)

    # Default shortcuts
    DEFAULT_SHORTCUTS = {
        "prev_item": "Left",
        "next_item": "Right",
        "save_current": "Ctrl+S",
        "save_and_next": "Ctrl+Shift+S",
        "prev_peak": "Up",
        "next_peak": "Down",
    }

    # Modifier key options for 3D viewer
    MODIFIER_OPTIONS = {
        "Alt": Qt.AltModifier,
        "Ctrl": Qt.ControlModifier,
        "Shift": Qt.ShiftModifier,
    }
    DEFAULT_MODIFIER = "Alt"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the preferences dialog."""
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(400)
        self._shortcut_edits: dict[str, QKeySequenceEdit] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)

        # Tab widget
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # Shortcuts tab
        shortcuts_tab = self._create_shortcuts_tab()
        self._tabs.addTab(shortcuts_tab, "Shortcuts")

        # Developer tab
        developer_tab = self._create_developer_tab()
        self._tabs.addTab(developer_tab, "Developer")

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )
        layout.addWidget(buttons)

    def _create_shortcuts_tab(self) -> QWidget:
        """Create the shortcuts configuration tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Navigation group
        nav_group = QGroupBox("Navigation")
        nav_layout = QFormLayout(nav_group)

        self._add_shortcut_row(nav_layout, "prev_item", "Previous Item:")
        self._add_shortcut_row(nav_layout, "next_item", "Next Item:")
        self._add_shortcut_row(nav_layout, "prev_peak", "Previous Peak:")
        self._add_shortcut_row(nav_layout, "next_peak", "Next Peak:")

        layout.addWidget(nav_group)

        # Actions group
        actions_group = QGroupBox("Actions")
        actions_layout = QFormLayout(actions_group)

        self._add_shortcut_row(actions_layout, "save_current", "Save Current:")
        self._add_shortcut_row(actions_layout, "save_and_next", "Save & Next:")

        layout.addWidget(actions_group)

        # 3D Viewer group
        viewer_group = QGroupBox("3D Viewer")
        viewer_layout = QFormLayout(viewer_group)

        # Modifier key for point size adjustment
        self._cmb_point_size_modifier = QComboBox()
        self._cmb_point_size_modifier.addItems(list(self.MODIFIER_OPTIONS.keys()))
        self._cmb_point_size_modifier.setCurrentText(self.DEFAULT_MODIFIER)
        self._cmb_point_size_modifier.setToolTip(
            "Modifier key + scroll wheel to adjust point cloud size (1-10)"
        )
        viewer_layout.addRow("Point Size Modifier:", self._cmb_point_size_modifier)

        # Description
        viewer_desc = QLabel(
            "<i>Hold modifier key + scroll wheel to change point cloud size.</i>"
        )
        viewer_desc.setWordWrap(True)
        viewer_desc.setStyleSheet("color: gray; margin-top: 4px;")
        viewer_layout.addRow("", viewer_desc)

        layout.addWidget(viewer_group)
        layout.addStretch()

        return widget

    def _create_developer_tab(self) -> QWidget:
        """Create the developer options tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Developer Mode group
        dev_group = QGroupBox("Developer Mode")
        dev_layout = QVBoxLayout(dev_group)

        # Debug logging checkbox
        self._chk_debug_log = QCheckBox("Enable Debug Logging")
        self._chk_debug_log.setToolTip(
            "When enabled, detailed debug messages will be shown in the console.\n"
            "This includes COCO loading details, pin segmentation steps, etc."
        )
        dev_layout.addWidget(self._chk_debug_log)

        # Description label
        desc_label = QLabel(
            "<i>Debug logging outputs detailed information to the console, "
            "useful for troubleshooting data loading and alignment issues.</i>"
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: gray; margin-top: 8px;")
        dev_layout.addWidget(desc_label)

        layout.addWidget(dev_group)
        layout.addStretch()

        return widget

    def _add_shortcut_row(
        self,
        layout: QFormLayout,
        action_id: str,
        label: str,
    ) -> None:
        """Add a shortcut edit row to the form."""
        edit = QKeySequenceEdit()
        default = self.DEFAULT_SHORTCUTS.get(action_id, "")
        edit.setKeySequence(QKeySequence(default))
        self._shortcut_edits[action_id] = edit
        layout.addRow(label, edit)

    def _restore_defaults(self) -> None:
        """Restore all settings to defaults."""
        # Restore shortcuts
        for action_id, edit in self._shortcut_edits.items():
            default = self.DEFAULT_SHORTCUTS.get(action_id, "")
            edit.setKeySequence(QKeySequence(default))

        # Restore modifier key
        self._cmb_point_size_modifier.setCurrentText(self.DEFAULT_MODIFIER)

        # Restore developer settings
        self._chk_debug_log.setChecked(False)

    def get_shortcuts(self) -> dict[str, str]:
        """
        Get the configured shortcuts.

        Returns
        -------
        dict[str, str]
            Dictionary mapping action IDs to shortcut strings.
        """
        return {
            action_id: edit.keySequence().toString()
            for action_id, edit in self._shortcut_edits.items()
        }

    def set_shortcuts(self, shortcuts: dict[str, str]) -> None:
        """
        Set shortcut values.

        Parameters
        ----------
        shortcuts : dict[str, str]
            Dictionary mapping action IDs to shortcut strings.
        """
        for action_id, shortcut in shortcuts.items():
            if action_id in self._shortcut_edits:
                self._shortcut_edits[action_id].setKeySequence(
                    QKeySequence(shortcut)
                )

    def get_developer_mode(self) -> bool:
        """
        Get the developer mode (debug logging) setting.

        Returns
        -------
        bool
            True if debug logging is enabled.
        """
        return self._chk_debug_log.isChecked()

    def set_developer_mode(self, enabled: bool) -> None:
        """
        Set the developer mode (debug logging) setting.

        Parameters
        ----------
        enabled : bool
            True to enable debug logging.
        """
        self._chk_debug_log.setChecked(enabled)

    def get_point_size_modifier(self) -> Qt.KeyboardModifier:
        """
        Get the modifier key for 3D viewer point size adjustment.

        Returns
        -------
        Qt.KeyboardModifier
            The selected modifier key.
        """
        key_name = self._cmb_point_size_modifier.currentText()
        return self.MODIFIER_OPTIONS.get(key_name, Qt.AltModifier)

    def get_point_size_modifier_name(self) -> str:
        """
        Get the modifier key name as string.

        Returns
        -------
        str
            Modifier key name ("Alt", "Ctrl", or "Shift").
        """
        return self._cmb_point_size_modifier.currentText()

    def set_point_size_modifier(self, modifier_name: str) -> None:
        """
        Set the modifier key for point size adjustment.

        Parameters
        ----------
        modifier_name : str
            Modifier key name ("Alt", "Ctrl", or "Shift").
        """
        if modifier_name in self.MODIFIER_OPTIONS:
            self._cmb_point_size_modifier.setCurrentText(modifier_name)

