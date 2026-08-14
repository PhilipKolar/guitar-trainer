"""Fretboard map drawn with QPainter.

Geometry comes from :mod:`guitar_trainer.core.notes`, so changing tuning or fret
count needs no changes here. Fret spacing follows the real rule rather than being
evenly divided, which matters for recognising shapes you've learnt on the instrument.
"""

from __future__ import annotations

from enum import Enum

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..core.notes import (
    DOUBLE_INLAY_FRETS,
    INLAY_FRETS,
    NATURAL_PITCH_CLASSES,
    STANDARD,
    Tuning,
    fret_x_positions,
    pitch_class_name,
)
from . import theme


class LabelMode(Enum):
    """How much of the note map to reveal — the "optionally visible notes" control."""

    NONE = "None"
    NATURALS = "Naturals only"
    ALL = "All notes"


class FretboardWidget(QWidget):
    """Interactive fretboard. Emits ``position_clicked(string, fret)``."""

    position_clicked = Signal(int, int)

    #: Left/right/top/bottom padding around the board, in pixels.
    MARGIN_LEFT = 46
    MARGIN_RIGHT = 18
    MARGIN_TOP = 26
    MARGIN_BOTTOM = 26

    def __init__(self, tuning: Tuning = STANDARD, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tuning = tuning
        self._label_mode = LabelMode.NONE
        self._use_flats = False
        self._detected_pitch_class: int | None = None
        self._target_pitch_classes: set[int] = set()
        self._target_positions: set[tuple[int, int]] = set()
        self._marker_color = theme.TARGET

        self.setMinimumHeight(190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ---------------------------------------------------------------- state

    @property
    def tuning(self) -> Tuning:
        return self._tuning

    def set_tuning(self, tuning: Tuning) -> None:
        self._tuning = tuning
        self.update()

    def set_label_mode(self, mode: LabelMode) -> None:
        self._label_mode = mode
        self.update()

    def set_use_flats(self, use_flats: bool) -> None:
        self._use_flats = use_flats
        self.update()

    def set_detected_pitch_class(self, pitch_class: int | None) -> None:
        """Highlight every position matching what is currently being heard.

        A mono signal can't tell us which string was played, so all matching positions
        light up — which is also the more useful thing to see while learning the map.
        """
        if pitch_class is not None:
            pitch_class %= 12
        if pitch_class != self._detected_pitch_class:
            self._detected_pitch_class = pitch_class
            self.update()

    def set_target_pitch_classes(self, pitch_classes, *, color: QColor | None = None) -> None:
        """Mark where the current challenge can be played."""
        new = {pc % 12 for pc in pitch_classes}
        color = color or theme.TARGET
        if new != self._target_pitch_classes or color != self._marker_color:
            self._target_pitch_classes = new
            self._target_positions = {
                pos for pc in new for pos in self._tuning.positions_for_pitch_class(pc)
            }
            self._marker_color = color
            self.update()

    def clear_targets(self) -> None:
        self.set_target_pitch_classes(())

    # ------------------------------------------------------------- geometry

    def _board_rect(self) -> QRectF:
        return QRectF(
            self.MARGIN_LEFT,
            self.MARGIN_TOP,
            max(1.0, self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT),
            max(1.0, self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM),
        )

    def _fret_xs(self) -> list[float]:
        """Absolute x of the nut and each fret wire."""
        rect = self._board_rect()
        return [rect.left() + d * rect.width() for d in fret_x_positions(self._tuning.fret_count)]

    def _string_y(self, string: int) -> float:
        """y for a string index. Index 0 is the low string, drawn at the bottom."""
        rect = self._board_rect()
        count = self._tuning.string_count
        if count == 1:
            return rect.center().y()
        spacing = rect.height() / (count - 1)
        return rect.bottom() - string * spacing

    def _marker_center(self, string: int, fret: int) -> QPointF:
        """Centre of a fingering dot: between the wires, or on the nut for an open string."""
        xs = self._fret_xs()
        if fret == 0:
            x = xs[0] - 16
        else:
            x = (xs[fret - 1] + xs[fret]) / 2
        return QPointF(x, self._string_y(string))

    def _position_at(self, x: float, y: float) -> tuple[int, int] | None:
        xs = self._fret_xs()
        string = min(
            range(self._tuning.string_count),
            key=lambda s: abs(self._string_y(s) - y),
        )
        if abs(self._string_y(string) - y) > 22:
            return None
        if x < xs[0] - 4:
            return (string, 0)
        for fret in range(1, len(xs)):
            if x <= xs[fret]:
                return (string, fret)
        return None

    # -------------------------------------------------------------- events

    def mousePressEvent(self, event) -> None:
        pos = self._position_at(event.position().x(), event.position().y())
        if pos is not None:
            self.position_clicked.emit(*pos)
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), theme.BACKGROUND)

        rect = self._board_rect()
        xs = self._fret_xs()

        self._draw_board(painter, rect)
        self._draw_inlays(painter, rect, xs)
        self._draw_frets(painter, rect, xs)
        self._draw_strings(painter, rect)
        self._draw_fret_numbers(painter, rect, xs)
        self._draw_string_labels(painter)
        self._draw_markers(painter)
        painter.end()

    # ------------------------------------------------------------- drawing

    def _draw_board(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(theme.FRETBOARD_EDGE, 2))
        painter.setBrush(theme.FRETBOARD_WOOD)
        painter.drawRoundedRect(rect.adjusted(-2, -10, 2, 10), 4, 4)

    def _draw_inlays(self, painter: QPainter, rect: QRectF, xs: list[float]) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(theme.INLAY)
        radius = 5.0
        for fret in list(INLAY_FRETS) + list(DOUBLE_INLAY_FRETS):
            if fret > self._tuning.fret_count:
                continue
            x = (xs[fret - 1] + xs[fret]) / 2
            if fret in DOUBLE_INLAY_FRETS:
                offset = rect.height() / 4
                painter.drawEllipse(QPointF(x, rect.center().y() - offset), radius, radius)
                painter.drawEllipse(QPointF(x, rect.center().y() + offset), radius, radius)
            else:
                painter.drawEllipse(QPointF(x, rect.center().y()), radius, radius)

    def _draw_frets(self, painter: QPainter, rect: QRectF, xs: list[float]) -> None:
        top, bottom = rect.top() - 10, rect.bottom() + 10
        # The nut is thicker and a different colour to the fret wires.
        painter.setPen(QPen(theme.NUT, 6))
        painter.drawLine(QPointF(xs[0], top), QPointF(xs[0], bottom))

        painter.setPen(QPen(theme.FRET_WIRE, 2))
        for x in xs[1:]:
            painter.drawLine(QPointF(x, top), QPointF(x, bottom))

    def _draw_strings(self, painter: QPainter, rect: QRectF) -> None:
        count = self._tuning.string_count
        for string in range(count):
            # Low strings are visibly thicker, as on the instrument.
            width = 1.0 + 2.0 * (count - 1 - string) / max(1, count - 1)
            painter.setPen(QPen(theme.STRING, width))
            y = self._string_y(string)
            painter.drawLine(QPointF(rect.left() - 20, y), QPointF(rect.right(), y))

    def _draw_fret_numbers(self, painter: QPainter, rect: QRectF, xs: list[float]) -> None:
        painter.setPen(theme.TEXT_DIM)
        font = QFont(painter.font())
        font.setPointSize(8)
        painter.setFont(font)
        for fret in list(INLAY_FRETS) + list(DOUBLE_INLAY_FRETS):
            if fret > self._tuning.fret_count:
                continue
            x = (xs[fret - 1] + xs[fret]) / 2
            painter.drawText(
                QRectF(x - 14, rect.bottom() + 12, 28, 16),
                Qt.AlignmentFlag.AlignCenter,
                str(fret),
            )

    def _draw_string_labels(self, painter: QPainter) -> None:
        painter.setPen(theme.TEXT_DIM)
        font = QFont(painter.font())
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        for string in range(self._tuning.string_count):
            painter.drawText(
                QRectF(0, self._string_y(string) - 9, 30, 18),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                self._tuning.string_label(string),
            )

    def _should_label(self, pitch_class: int) -> bool:
        if self._label_mode is LabelMode.ALL:
            return True
        if self._label_mode is LabelMode.NATURALS:
            return pitch_class in NATURAL_PITCH_CLASSES
        return False

    def _draw_markers(self, painter: QPainter) -> None:
        font = QFont(painter.font())
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        radius = 10.0

        for string in range(self._tuning.string_count):
            for fret in range(self._tuning.fret_count + 1):
                pitch_class = self._tuning.midi_at(string, fret) % 12
                is_detected = pitch_class == self._detected_pitch_class
                is_target = (string, fret) in self._target_positions
                labelled = self._should_label(pitch_class)

                if not (is_detected or is_target or labelled):
                    continue

                center = self._marker_center(string, fret)

                if is_detected:
                    fill, text_color = theme.DETECTED, QColor("#08121e")
                elif is_target:
                    fill, text_color = self._marker_color, QColor("#1e1305")
                else:
                    fill, text_color = QColor(0, 0, 0, 130), theme.TEXT

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(fill)
                painter.drawEllipse(center, radius, radius)

                # A target that is also being played gets a ring, so a correct answer
                # reads at a glance without needing to compare two colours.
                if is_detected and is_target:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(self._marker_color, 2.5))
                    painter.drawEllipse(center, radius + 3, radius + 3)

                if labelled or is_detected or is_target:
                    painter.setPen(text_color)
                    painter.drawText(
                        QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2),
                        Qt.AlignmentFlag.AlignCenter,
                        pitch_class_name(pitch_class, use_flats=self._use_flats),
                    )
