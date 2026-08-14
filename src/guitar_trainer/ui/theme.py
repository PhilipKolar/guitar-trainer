"""Shared colours and fonts.

Kept in one place so the fretboard, tuner and practice panels stay visually
consistent, and so the palette can be adjusted without hunting through widgets.
"""

from __future__ import annotations

from PySide6.QtGui import QColor

BACKGROUND = QColor("#1b1d21")
PANEL = QColor("#24272c")
TEXT = QColor("#e6e8ea")
TEXT_DIM = QColor("#8b9199")

FRETBOARD_WOOD = QColor("#3a2a20")
FRETBOARD_EDGE = QColor("#14100d")
FRET_WIRE = QColor("#a9adb3")
NUT = QColor("#e8e4dc")
INLAY = QColor("#cfc9bd")
STRING = QColor("#c8ccd2")

#: Positions matching the note currently being heard.
DETECTED = QColor("#4da3ff")
#: Positions where the current challenge can be played.
TARGET = QColor("#ffb340")
CORRECT = QColor("#4ade80")
INCORRECT = QColor("#f87171")
IN_TUNE = QColor("#4ade80")
OUT_OF_TUNE = QColor("#f0a742")

STYLESHEET = f"""
QWidget {{
    background-color: {BACKGROUND.name()};
    color: {TEXT.name()};
    font-size: 13px;
}}
QGroupBox {{
    border: 1px solid #34383e;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    color: {TEXT_DIM.name()};
}}
QPushButton {{
    background-color: {PANEL.name()};
    border: 1px solid #3c4149;
    border-radius: 5px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background-color: #2c3037; }}
QPushButton:pressed {{ background-color: #1f2227; }}
QPushButton:checked {{
    background-color: #2f4a6b;
    border-color: {DETECTED.name()};
}}
QPushButton:disabled {{ color: #5a6068; }}
QComboBox, QSpinBox, QDoubleSpinBox {{
    background-color: {PANEL.name()};
    border: 1px solid #3c4149;
    border-radius: 5px;
    padding: 4px 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {PANEL.name()};
    selection-background-color: #2f4a6b;
}}
QTabWidget::pane {{ border: 1px solid #34383e; border-radius: 6px; }}
QTabBar::tab {{
    background: {BACKGROUND.name()};
    padding: 8px 18px;
    border: 1px solid #34383e;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}}
QTabBar::tab:selected {{ background: {PANEL.name()}; color: {DETECTED.name()}; }}
QTableWidget {{
    background-color: {PANEL.name()};
    gridline-color: #34383e;
    border: 1px solid #34383e;
    border-radius: 6px;
}}
QHeaderView::section {{
    background-color: {BACKGROUND.name()};
    border: none;
    padding: 6px;
    color: {TEXT_DIM.name()};
}}
QCheckBox, QRadioButton {{ padding: 2px; }}
"""
