# -*- coding: utf-8 -*-
"""
File tree widget with checkbox status for registration files.

Displays available potato IDs with checkboxes indicating completion status.
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class FileTreeWidget(QWidget):
    """
    File tree widget showing potato IDs with completion status.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.

    Signals
    -------
    item_selected : Signal(str)
        Emitted when an item is selected. Carries the potato ID.
    item_double_clicked : Signal(str)
        Emitted when an item is double-clicked.

    Examples
    --------
    >>> tree = FileTreeWidget()
    >>> tree.set_items(["2R1-1", "2R1-2"], completed=["2R1-1"])
    >>> tree.item_selected.connect(on_item_selected)
    """

    item_selected = Signal(str)
    item_double_clicked = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the file tree widget."""
        super().__init__(parent)
        self._items: dict[str, QTreeWidgetItem] = {}
        self._completed: set[str] = set()
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Status", "ID"])
        self._tree.setColumnCount(2)
        self._tree.setRootIsDecorated(False)
        self._tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tree.setAlternatingRowColors(True)

        # Column widths
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)

        # Connect signals
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.itemDoubleClicked.connect(self._on_double_clicked)

        layout.addWidget(self._tree)

    def set_items(
        self,
        ids: list[str],
        completed: Optional[list[str]] = None,
    ) -> None:
        """
        Set the list of potato IDs.

        Parameters
        ----------
        ids : list[str]
            List of all potato IDs.
        completed : list[str], optional
            List of completed (aligned) IDs.
        """
        self._tree.clear()
        self._items.clear()
        self._completed = set(completed or [])

        for pid in ids:
            item = QTreeWidgetItem()
            is_done = pid in self._completed

            # Status column
            item.setText(0, "✓" if is_done else "○")
            if is_done:
                item.setForeground(0, QBrush(QColor("#4CAF50")))
            else:
                item.setForeground(0, QBrush(QColor("#9E9E9E")))

            # ID column
            item.setText(1, pid)
            item.setData(1, Qt.UserRole, pid)

            self._tree.addTopLevelItem(item)
            self._items[pid] = item

    def set_completed(self, pid: str, completed: bool = True) -> None:
        """
        Update completion status for an ID.

        Parameters
        ----------
        pid : str
            Potato ID to update.
        completed : bool
            Whether the ID is completed.
        """
        if pid not in self._items:
            return

        item = self._items[pid]
        if completed:
            self._completed.add(pid)
            item.setText(0, "✓")
            item.setForeground(0, QBrush(QColor("#4CAF50")))
        else:
            self._completed.discard(pid)
            item.setText(0, "○")
            item.setForeground(0, QBrush(QColor("#9E9E9E")))

    def select_item(self, pid: str) -> None:
        """
        Select an item by ID.

        Parameters
        ----------
        pid : str
            Potato ID to select.
        """
        if pid in self._items:
            self._tree.setCurrentItem(self._items[pid])

    def get_selected_id(self) -> Optional[str]:
        """
        Get the currently selected potato ID.

        Returns
        -------
        str or None
            Selected ID, or None if nothing selected.
        """
        items = self._tree.selectedItems()
        if items:
            return items[0].data(1, Qt.UserRole)
        return None

    def get_next_uncompleted(self) -> Optional[str]:
        """
        Get the next uncompleted potato ID.

        Returns
        -------
        str or None
            Next uncompleted ID, or None if all completed.
        """
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            pid = item.data(1, Qt.UserRole)
            if pid not in self._completed:
                return pid
        return None

    def select_next_uncompleted(self) -> bool:
        """
        Select the next uncompleted item.

        Returns
        -------
        bool
            True if an item was selected, False if all completed.
        """
        next_id = self.get_next_uncompleted()
        if next_id:
            self.select_item(next_id)
            return True
        return False

    @Slot()
    def _on_selection_changed(self) -> None:
        """Handle selection change in tree."""
        pid = self.get_selected_id()
        if pid:
            self.item_selected.emit(pid)

    @Slot(QTreeWidgetItem, int)
    def _on_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle double-click on tree item."""
        pid = item.data(1, Qt.UserRole)
        if pid:
            self.item_double_clicked.emit(pid)
