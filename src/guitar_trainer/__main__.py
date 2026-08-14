"""Application entry point."""

from __future__ import annotations

import signal
import sys


def main() -> int:
    from PySide6.QtCore import QTimer
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

    # Qt swallows SIGINT by default, so Ctrl-C in the launching terminal — the normal
    # way to stop something started with run.sh — just kills the process outright,
    # skipping closeEvent (and the config save it triggers) entirely. Route it through
    # a graceful window close instead.
    signal.signal(signal.SIGINT, lambda *_: window.close())
    # Python only gets a chance to run that handler between bytecode instructions,
    # and Qt's event loop otherwise blocks in C++ without ever yielding back to the
    # interpreter. A trivial periodic timer is what gives it that chance promptly.
    keepalive = QTimer()
    keepalive.timeout.connect(lambda: None)
    keepalive.start(200)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
