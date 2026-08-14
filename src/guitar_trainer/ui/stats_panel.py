"""Practice history view."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.stats import StatsStore
from . import theme


class StatsPanel(QWidget):
    """Per-note and per-chord accuracy and response times."""

    def __init__(self, store: StatsStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store

        self.summary = QLabel("")
        self.summary.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Challenge", "Attempts", "Accuracy", "Median time"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        reset = QPushButton("Clear history")
        reset.clicked.connect(self._confirm_reset)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(refresh)
        buttons.addWidget(reset)

        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(self.table)
        layout.addLayout(buttons)

        self.refresh()

    def refresh(self) -> None:
        attempts, correct = self.store.totals()
        sessions = self.store.session_count()
        if attempts:
            self.summary.setText(
                f"{sessions} session{'s' if sessions != 1 else ''} · "
                f"{correct}/{attempts} correct ({correct / attempts:.0%})"
            )
        else:
            self.summary.setText("No practice history yet")

        stats = self.store.challenge_stats()
        # Weakest first: that's what you'd open this panel to find.
        stats.sort(key=lambda s: (s.accuracy, -(s.median_response_ms or 0)))

        self.table.setRowCount(len(stats))
        for row, entry in enumerate(stats):
            accuracy_item = QTableWidgetItem(f"{entry.accuracy:.0%}")
            if entry.accuracy >= 0.9:
                accuracy_item.setForeground(theme.CORRECT)
            elif entry.accuracy < 0.6:
                accuracy_item.setForeground(theme.INCORRECT)

            median = entry.median_response_ms
            cells = [
                QTableWidgetItem(entry.prompt),
                QTableWidgetItem(str(entry.attempts)),
                accuracy_item,
                QTableWidgetItem(f"{median / 1000:.1f}s" if median else "—"),
            ]
            for column, item in enumerate(cells):
                if column:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)

    def _confirm_reset(self) -> None:
        answer = QMessageBox.question(
            self,
            "Clear history",
            "Delete all practice history? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.store.reset()
            self.refresh()
