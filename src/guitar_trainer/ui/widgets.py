"""Small shared widgets."""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt
from PySide6.QtGui import QFont, QLinearGradient, QPainter
from PySide6.QtWidgets import QGraphicsOpacityEffect, QLabel, QSizePolicy, QVBoxLayout, QWidget

from . import theme


class LevelMeter(QWidget):
    """Input level bar with a marker showing where the noise gate sits.

    Setting the gate blind is guesswork, so the threshold is drawn on the same scale as
    the signal: put the marker below where playing peaks and above where the room sits.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._level = 0.0
        self._peak = 0.0
        self._threshold = 0.01
        self.setFixedHeight(14)
        self.setMinimumWidth(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_level(self, rms: float) -> None:
        self._level = max(0.0, min(1.0, rms * 4.0))
        # Peak falls back slowly so brief transients stay readable.
        self._peak = max(self._level, self._peak * 0.94)
        self.update()

    def set_threshold(self, threshold: float) -> None:
        self._threshold = threshold
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(theme.PANEL)
        painter.drawRoundedRect(rect, 3, 3)

        if self._level > 0:
            gradient = QLinearGradient(rect.left(), 0, rect.right(), 0)
            gradient.setColorAt(0.0, theme.IN_TUNE)
            gradient.setColorAt(0.7, theme.OUT_OF_TUNE)
            gradient.setColorAt(1.0, theme.INCORRECT)
            painter.setBrush(gradient)
            filled = QRectF(rect)
            filled.setWidth(rect.width() * self._level)
            painter.drawRoundedRect(filled, 3, 3)

        if self._peak > 0.01:
            x = rect.left() + rect.width() * self._peak
            painter.setPen(theme.TEXT)
            painter.drawLine(x, rect.top() + 2, x, rect.bottom() - 2)

        # Gate marker, on the same scale as the level.
        x = rect.left() + rect.width() * min(1.0, self._threshold * 4.0)
        painter.setPen(theme.TEXT_DIM)
        painter.drawLine(x, rect.top(), x, rect.bottom())
        painter.end()


class CentsMeter(QWidget):
    """Needle showing how sharp or flat the current note is."""

    RANGE_CENTS = 50.0
    IN_TUNE_CENTS = 5.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cents: float | None = None
        self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_cents(self, cents: float | None) -> None:
        self._cents = cents
        self.update()

    @property
    def in_tune(self) -> bool:
        return self._cents is not None and abs(self._cents) <= self.IN_TUNE_CENTS

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        center_x = rect.center().x()
        scale_y = rect.top() + 34

        font = QFont(painter.font())
        font.setPointSize(8)
        painter.setFont(font)

        # Scale: a tick every 10 cents, labelled at the extremes and centre.
        for cents in range(-50, 51, 10):
            x = center_x + (cents / self.RANGE_CENTS) * (rect.width() / 2 - 20)
            is_major = cents in (-50, 0, 50)
            painter.setPen(theme.TEXT if is_major else theme.TEXT_DIM)
            height = 10 if is_major else 5
            painter.drawLine(x, scale_y - height, x, scale_y + height)
            if is_major:
                painter.drawText(
                    QRectF(x - 20, scale_y + 12, 40, 14),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{cents:+d}" if cents else "0",
                )

        if self._cents is None:
            painter.end()
            return

        clamped = max(-self.RANGE_CENTS, min(self.RANGE_CENTS, self._cents))
        x = center_x + (clamped / self.RANGE_CENTS) * (rect.width() / 2 - 20)
        color = theme.IN_TUNE if self.in_tune else theme.OUT_OF_TUNE

        pen = painter.pen()
        pen.setColor(color)
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawLine(x, rect.top() + 2, x, scale_y + 12)
        painter.end()


class BigNoteLabel(QLabel):
    """The large note or chord readout."""

    def __init__(self, placeholder: str = "—", parent: QWidget | None = None) -> None:
        super().__init__(placeholder, parent)
        self.placeholder = placeholder
        font = QFont(self.font())
        font.setPointSize(64)
        font.setBold(True)
        self.setFont(font)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(100)

    def show_text(self, text: str | None, color=None) -> None:
        self.setText(text or self.placeholder)
        color = color or (theme.TEXT if text else theme.TEXT_DIM)
        self.setStyleSheet(f"color: {color.name()};")


class CaptionedLabel(QWidget):
    """A small caption above a value — the previous/next challenge preview."""

    def __init__(self, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.caption = QLabel(caption)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        caption_font = QFont(self.caption.font())
        caption_font.setPointSize(9)
        self.caption.setFont(caption_font)
        self.caption.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")

        self.value = QLabel("—")
        self.value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_font = QFont(self.value.font())
        value_font.setPointSize(22)
        value_font.setBold(True)
        self.value.setFont(value_font)
        self.value.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.caption)
        layout.addWidget(self.value)

    def set_text(self, text: str | None) -> None:
        self.value.setText(text or "—")


def fade_in(widget: QWidget, *, duration_ms: int = 220) -> QPropertyAnimation:
    """Attach a reusable fade-from-dim animation to a widget and return it.

    Call ``.stop(); .start()`` on the result whenever the widget's content changes, so
    each change gets a brief, tasteful pop rather than appearing instantly. Keeping the
    QGraphicsOpacityEffect and QPropertyAnimation alive as attributes on the widget's
    parent is the caller's responsibility — Qt does not keep them alive on its own.
    """
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    animation = QPropertyAnimation(effect, b"opacity", widget)
    animation.setDuration(duration_ms)
    animation.setStartValue(0.3)
    animation.setEndValue(1.0)
    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    return animation
