# -*- coding: utf-8 -*-
"""
Preferences dialog with shortcut editing.

Allows users to configure keyboard shortcuts for common actions.
"""

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QKeySequenceEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class PreferencesDialog(QDialog):
    """
    Preferences dialog for shortcut configuration.

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
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Shortcuts tab
        shortcuts_tab = self._create_shortcuts_tab()
        tabs.addTab(shortcuts_tab, "Shortcuts")

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
        """Restore all shortcuts to defaults."""
        for action_id, edit in self._shortcut_edits.items():
            default = self.DEFAULT_SHORTCUTS.get(action_id, "")
            edit.setKeySequence(QKeySequence(default))

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
