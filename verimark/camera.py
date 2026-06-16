"""IP kamera baglantisi (telefon IP kamera uygulamasi / RTSP / webcam).

Ayri bir thread surekli en guncel kareyi okur; boylece MJPEG buffer
gecikmesi olusmaz ve UI her zaman taze kare alir.
"""

from __future__ import annotations

import threading
import time

import cv2


class IPCamera:
    def __init__(self, url: str):
        # "0", "1" gibi yerel kamera indeksleri destegi
        self.url: object = int(url) if str(url).isdigit() else url
        self.cap = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._frame = None
        self._last_ts = 0.0
        self.connected = False
        self.error = ""

    def open(self, timeout: float = 6.0) -> bool:
        self.cap = cv2.VideoCapture(self.url)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.cap.isOpened():
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    with self._lock:
                        self._frame = frame
                        self._last_ts = time.time()
                    self.connected = True
                    break
            time.sleep(0.1)

        if not self.connected:
            self.error = "Baglanti kurulamadi. URL / IP / ag baglantisini kontrol edin."
            self.release()
            return False

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        fail = 0
        while self._running:
            if self.cap is None:
                break
            ok, frame = self.cap.read()
            if not ok or frame is None:
                fail += 1
                self.connected = False
                time.sleep(0.05)
                if fail % 20 == 0:  # periyodik yeniden baglanma denemesi
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = cv2.VideoCapture(self.url)
                continue
            fail = 0
            self.connected = True
            with self._lock:
                self._frame = frame
                self._last_ts = time.time()

    def read(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    @property
    def age(self) -> float:
        """Son karenin yasi (saniye)."""
        return time.time() - self._last_ts if self._last_ts else 1e9

    def release(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.connected = False
