"""Goruntu gosterme + fare ile ROI (dikdortgen) secme widget'i.

ROI, goruntu (piksel) koordinatlarinda dondurulur; ekran olceginden bagimsiz.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt6.QtWidgets import QWidget


class ImageView(QWidget):
    roiSelected = pyqtSignal(tuple)  # (x, y, w, h) goruntu koordinatlarinda

    def __init__(self, parent=None, selectable: bool = False):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self._frame: np.ndarray | None = None
        self._pix: QPixmap | None = None
        self._selectable = selectable
        self._draw_scale = 1.0
        self._off = QPoint(0, 0)
        self._disp_size = (0, 0)
        self._dragging = False
        self._p0 = QPoint()
        self._p1 = QPoint()
        self._roi: tuple[int, int, int, int] | None = None
        self.setMouseTracking(True)

    # -- API --
    def set_frame(self, bgr: np.ndarray | None):
        self._frame = bgr
        if bgr is None:
            self._pix = None
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self._pix = QPixmap.fromImage(qimg.copy())
        self.update()

    def set_selectable(self, val: bool):
        self._selectable = val
        if not val:
            self._roi = None
        self.update()

    def clear_roi(self):
        self._roi = None
        self.update()

    @property
    def roi(self):
        return self._roi

    # -- ciktilar --
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 22))
        if self._pix is None:
            p.setPen(QColor(160, 160, 160))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Goruntu yok\n(kameraya baglanin)")
            return

        iw, ih = self._pix.width(), self._pix.height()
        scale = min(self.width() / iw, self.height() / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        ox, oy = (self.width() - dw) // 2, (self.height() - dh) // 2
        self._draw_scale = scale
        self._off = QPoint(ox, oy)
        self._disp_size = (dw, dh)
        p.drawPixmap(ox, oy, dw, dh, self._pix)

        # secim dikdortgeni
        if self._selectable and (self._dragging or self._roi is not None):
            p.setPen(QPen(QColor(0, 220, 120), 2, Qt.PenStyle.DashLine))
            if self._dragging:
                r = QRect(self._p0, self._p1).normalized()
            else:
                x, y, w, h = self._roi
                r = QRect(ox + int(x * scale), oy + int(y * scale),
                          int(w * scale), int(h * scale))
            p.drawRect(r)

    # -- fare olaylari (sadece secilebilir modda) --
    def mousePressEvent(self, e):
        if not self._selectable or self._pix is None:
            return
        self._dragging = True
        self._p0 = e.position().toPoint()
        self._p1 = self._p0

    def mouseMoveEvent(self, e):
        if self._dragging:
            self._p1 = e.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, e):
        if not self._dragging:
            return
        self._dragging = False
        self._p1 = e.position().toPoint()
        roi = self._to_image_roi(QRect(self._p0, self._p1).normalized())
        if roi and roi[2] > 8 and roi[3] > 8:
            self._roi = roi
            self.roiSelected.emit(roi)
        self.update()

    def _to_image_roi(self, r: QRect):
        if self._pix is None or self._draw_scale == 0:
            return None
        ox, oy = self._off.x(), self._off.y()
        iw, ih = self._pix.width(), self._pix.height()
        x = (r.x() - ox) / self._draw_scale
        y = (r.y() - oy) / self._draw_scale
        w = r.width() / self._draw_scale
        h = r.height() / self._draw_scale
        x = max(0, min(iw - 1, x))
        y = max(0, min(ih - 1, y))
        w = max(1, min(iw - x, w))
        h = max(1, min(ih - y, h))
        return (int(x), int(y), int(w), int(h))
