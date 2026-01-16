# -*- coding: utf-8 -*-
"""
Application entry point for the SFM-RGBD Registration GUI.

Usage
-----
Run from the project root:
    python -m 03_sfm_rgbd_registration_qt.main

Or directly:
    python 03_sfm_rgbd_registration_qt/main.py
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Force X11 backend for Qt/VTK compatibility on Wayland
# Must be set BEFORE importing PySide6
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["XDG_SESSION_TYPE"] = "x11"

# Ensure package imports work
sys.path.insert(0, str(Path(__file__).parent))

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> int:
    """
    Application entry point.

    Returns
    -------
    int
        Exit code.
    """
    app = QApplication(sys.argv)
    app.setApplicationName("SFM-RGBD Registration Tool")
    app.setOrganizationName("PotatoScan")
    app.setOrganizationDomain("github.com/PotatoScan")

    # Set application style
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
