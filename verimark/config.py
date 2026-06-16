"""Yapilandirma: esikler, kamera onayarlari, dosya yollari."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict, field

# ---------------------------------------------------------------------------
# Dosya yollari
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _data_dir() -> str:
    """Profillerin yazilacagi klasor.

    Kaynaktan calistirilirken proje koku kullanilir (gelistirici rahatligi).
    PyInstaller ile paketlenmis .exe'de (sys.frozen) kurulum klasoru genelde
    salt-okunur ( or. Program Files) oldugundan kullaniciya ozel yazilabilir
    bir klasore yazilir:
      Windows : %LOCALAPPDATA%\\VeriMark
      macOS   : ~/Library/Application Support/VeriMark
      Linux   : ~/.local/share/VeriMark
    """
    if getattr(sys, "frozen", False):
        if sys.platform.startswith("win"):
            base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        elif sys.platform == "darwin":
            base = os.path.expanduser("~/Library/Application Support")
        else:
            base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
        return os.path.join(base, "VeriMark")
    return ROOT_DIR


DATA_DIR = _data_dir()
PROFILES_DIR = os.path.join(DATA_DIR, "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Kamera onayarlari (telefon IP kamera uygulamalari)
# {ip} otomatik olarak girilen IP ile degistirilir.
# ---------------------------------------------------------------------------
CAMERA_PRESETS = {
    "IP Webcam (Android) - video": "http://{ip}:8080/video",
    "IP Webcam (Android) - shot": "http://{ip}:8080/shot.jpg",
    "DroidCam": "http://{ip}:4747/video",
    "RTSP (genel)": "rtsp://{ip}:554/stream",
    "Webcam (yerel 0)": "0",
}

DEFAULT_IP = "192.168.1.100"


@dataclass
class Thresholds:
    """Denetim esikleri. Hassasiyet kaydiricisi bunlari olceklendirir."""

    min_matches: int = 10            # Logo bulundu sayilmasi icin min ozellik eslesmesi
    ssim_min: float = 0.62           # Yapisal benzerlik alt siniri (0..1)
    ink_tolerance: float = 0.18      # Murekkep kapsama orani sapma toleransi (bagil)
    deltaE_max: float = 12.0         # Renk/ton sapmasi (CIEDE2000)
    rotation_max_deg: float = 8.0    # Izin verilen donme (derece)
    scale_tolerance: float = 0.15    # Izin verilen olcek sapmasi (bagil)
    defect_area_frac: float = 0.0010 # Kusur sayilmasi icin min alan (ROI'ye orani)
    defect_diff_level: float = 0.34  # Fark haritasi esigi (0..1)
    max_defects: int = 0             # Izin verilen buyuk kusur sayisi
    anomaly_max: float = 1.0         # Derin anomali skoru ust siniri (1.0 = kalibre esik)
    anomaly_heat_level: float = 0.5  # Anomali isi haritasi kusur esigi (0..1)

    def asdict(self) -> dict:
        return asdict(self)

    @classmethod
    def fromdict(cls, d: dict) -> "Thresholds":
        known = {k: v for k, v in (d or {}).items() if k in cls.__annotations__}
        return cls(**known)

    def scaled(self, sensitivity: float) -> "Thresholds":
        """Hassasiyet 0..100. 50 = varsayilan. Yuksek = daha siki denetim."""
        s = max(0.0, min(100.0, sensitivity))
        k = (s - 50.0) / 50.0  # -1..+1
        return Thresholds(
            min_matches=self.min_matches,
            ssim_min=min(0.97, self.ssim_min + 0.18 * k),
            ink_tolerance=max(0.03, self.ink_tolerance - 0.10 * k),
            deltaE_max=max(3.0, self.deltaE_max - 7.0 * k),
            rotation_max_deg=max(2.0, self.rotation_max_deg - 5.0 * k),
            scale_tolerance=max(0.04, self.scale_tolerance - 0.10 * k),
            defect_area_frac=max(0.0002, self.defect_area_frac - 0.0007 * k),
            defect_diff_level=max(0.15, self.defect_diff_level - 0.16 * k),
            max_defects=self.max_defects,
            anomaly_max=max(0.6, self.anomaly_max - 1.0 * k),
            anomaly_heat_level=max(0.25, self.anomaly_heat_level - 0.25 * k),
        )
