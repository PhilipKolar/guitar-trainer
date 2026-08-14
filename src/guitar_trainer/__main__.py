"""Application entry point."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    from .ui import theme
    from .ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Guitar Trainer")

    font = QFont(app.font())
    font.setPointSize(theme.BASE_FONT_POINT_SIZE)
    app.setFont(font)
    app.setStyleSheet(theme.STYLESHEET)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
