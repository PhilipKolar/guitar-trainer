"""Shared colours and fonts.

Kept in one place so the fretboard, tuner and practice panels stay visually
consistent, and so the palette can be adjusted without hunting through widgets.
"""

from __future__ import annotations

from pathlib import Path

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

#: Base UI font size. Applied as the application font rather than through the
#: stylesheet: a `font-size` rule in a stylesheet overrides any `setFont` on a widget,
#: which would silently flatten the large note and prompt readouts back to body text.
BASE_FONT_POINT_SIZE = 10


def _icon_dir() -> Path:
    from ..core.config import data_dir

    path = data_dir() / "icons"
    path.mkdir(parents=True, exist_ok=True)
    return path


#: (width, height, polygon points, colour) for each arrow icon, keyed by filename.
_ARROW_SPECS: dict[str, tuple[int, int, list[tuple[float, float]], QColor]] = {
    "combo-arrow.png": (10, 6, [(0, 0), (10, 0), (5, 6)], TEXT_DIM),
    "spin-up-arrow.png": (8, 5, [(0, 5), (8, 5), (4, 0)], TEXT_DIM),
    "spin-down-arrow.png": (8, 5, [(0, 0), (8, 0), (4, 5)], TEXT_DIM),
    "spin-up-arrow-disabled.png": (8, 5, [(0, 5), (8, 5), (4, 0)], QColor("#454b53")),
    "spin-down-arrow-disabled.png": (8, 5, [(0, 0), (8, 0), (4, 5)], QColor("#454b53")),
}


def _ensure_arrow_icons() -> dict[str, Path]:
    """Render the combo/spin-box arrow triangles to small cached PNGs.

    Qt's usual QSS trick for a custom arrow — a zero-size box with transparent side
    borders and a coloured top/bottom border, forming a triangle — does not render
    reliably here; it shows as a flat bar instead of a triangle. Drawing real pixmaps
    is what actually works, and is also what fixes the original complaint: native
    drop-down and spin-box arrows render via different code paths and come out looking
    like two different control families on a dark palette. One drawing routine, one
    colour, used for both, fixes both problems at once. Cached under the app's data
    directory so this only runs once per install rather than every launch.
    """
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QPainter, QPixmap, QPolygonF

    icon_dir = _icon_dir()
    paths: dict[str, Path] = {}
    for filename, (width, height, points, color) in _ARROW_SPECS.items():
        path = icon_dir / filename
        paths[filename] = path
        if path.exists():
            continue
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
        painter.end()
        pixmap.save(str(path))
    return paths


def stylesheet() -> str:
    """The application stylesheet. Must be called after a QApplication exists —
    generating the arrow icons requires one."""
    icons = _ensure_arrow_icons()

    def url(filename: str) -> str:
        return icons[filename].as_posix()

    return f"""
QWidget {{
    background-color: {BACKGROUND.name()};
    color: {TEXT.name()};
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
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: #4a5058; }}
QComboBox QAbstractItemView {{
    background-color: {PANEL.name()};
    selection-background-color: #2f4a6b;
}}

/* Native drop-down/spin arrows render inconsistently across widget types on a dark
   palette (mismatched colours, poor contrast) — drawn explicitly here instead, the
   same triangle style and colour for both, so a combo box and a spin box read as one
   family of control rather than two different-looking ones. */
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid #3c4149;
    border-top-right-radius: 5px;
    border-bottom-right-radius: 5px;
    background-color: {PANEL.name()};
}}
QComboBox::drop-down:hover {{ background-color: #2c3037; }}
QComboBox::down-arrow {{
    image: url({url("combo-arrow.png")});
    width: 10px;
    height: 6px;
    margin-right: 7px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid #3c4149;
    border-bottom: 1px solid #3c4149;
    border-top-right-radius: 5px;
    background-color: {PANEL.name()};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid #3c4149;
    border-bottom-right-radius: 5px;
    background-color: {PANEL.name()};
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: #2c3037;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url({url("spin-up-arrow.png")});
    width: 8px;
    height: 5px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({url("spin-down-arrow.png")});
    width: 8px;
    height: 5px;
}}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{
    image: url({url("spin-up-arrow-disabled.png")});
}}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
    image: url({url("spin-down-arrow-disabled.png")});
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
