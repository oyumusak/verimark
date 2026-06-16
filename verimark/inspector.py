"""Denetim cekirdegi: referans model + karsilastirma metrikleri + karar.

Olculen metrikler:
  - SSIM (yapisal benzerlik)        -> soluk/eksik baski, genel bozulma
  - Murekkep kapsama orani          -> soluk / eksik / fazla murekkep
  - Renk sapmasi (CIEDE2000)        -> ton/renk hatasi
  - Donme / olcek (homografiden)    -> kayma/donme/olcek hatasi
  - Kusur blob'lari (fark haritasi) -> leke / cizik / lokal hata
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity as ssim

from .alignment import Aligner, to_gray
from .config import Thresholds


# ---------------------------------------------------------------------------
# Referans model
# ---------------------------------------------------------------------------
class ReferenceModel:
    """Ogrenilmis logonun referans verileri."""

    def __init__(self, ref_bgr: np.ndarray):
        self.ref_bgr = ref_bgr
        self.h, self.w = ref_bgr.shape[:2]
        self.ref_gray = to_gray(ref_bgr)

        aligner = Aligner()
        self.kp, self.des = aligner.detect(ref_bgr)

        self.ink_mask = _ink_mask(ref_bgr)
        self.ref_coverage = float(self.ink_mask.mean() / 255.0)
        self.ref_lab = _mean_lab(ref_bgr, self.ink_mask)

    @property
    def size(self) -> tuple[int, int]:
        return (self.w, self.h)


# ---------------------------------------------------------------------------
# Denetim sonucu
# ---------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    value: float
    limit: float
    passed: bool
    detail: str = ""


@dataclass
class InspectResult:
    ok: bool
    found: bool
    message: str
    checks: list[Check] = field(default_factory=list)
    aligned: np.ndarray | None = None
    heatmap: np.ndarray | None = None
    defect_boxes: list = field(default_factory=list)
    frame_box: np.ndarray | None = None     # logonun ham karedeki konumu
    rotation_deg: float = 0.0
    scale: float = 1.0
    n_matches: int = 0


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------
class Inspector:
    def __init__(self, model: ReferenceModel, thresholds: Thresholds):
        self.model = model
        self.th = thresholds
        self.aligner = Aligner()

    def inspect(self, frame: np.ndarray) -> InspectResult:
        m = self.model
        th = self.th

        al = self.aligner.align(frame, m.kp, m.des, m.size, th.min_matches)
        if not al.ok:
            return InspectResult(
                ok=False, found=False, message=al.reason,
                n_matches=al.n_matches,
            )

        aligned = al.aligned
        aligned_gray = to_gray(aligned)

        # --- SSIM ---
        score, diff = ssim(m.ref_gray, aligned_gray, full=True)
        diffmap = (1.0 - diff)  # 0..1, yuksek = farkli

        # --- Murekkep kapsama ---
        cand_mask = _ink_mask(aligned)
        cand_cov = float(cand_mask.mean() / 255.0)
        cov_dev = abs(cand_cov - m.ref_coverage) / max(1e-4, m.ref_coverage)

        # --- Renk (CIEDE2000) --- sadece logo (on plan) bolgesinde
        cand_lab = _mean_lab(aligned, m.ink_mask)
        dE = float(deltaE_ciede2000(m.ref_lab[None, :], cand_lab[None, :])[0])

        # --- Kusur blob'lari ---
        boxes, heat = _find_defects(diffmap, m.ref_gray, th)
        roi_area = m.w * m.h
        big_defects = [b for b in boxes if b[2] * b[3] >= th.defect_area_frac * roi_area]

        # --- Kararlar ---
        checks = [
            Check("Yapisal benzerlik (SSIM)", round(score, 3), th.ssim_min,
                  score >= th.ssim_min, "Baski netligi/butunlugu"),
            Check("Murekkep kapsama sapmasi", round(cov_dev, 3), th.ink_tolerance,
                  cov_dev <= th.ink_tolerance, "Soluk / eksik / fazla murekkep"),
            Check("Renk sapmasi (dE2000)", round(dE, 2), th.deltaE_max,
                  dE <= th.deltaE_max, "Ton / renk farki"),
            Check("Donme (derece)", round(abs(al.rotation_deg), 2), th.rotation_max_deg,
                  abs(al.rotation_deg) <= th.rotation_max_deg, "Aci kaymasi"),
            Check("Olcek sapmasi", round(abs(al.scale - 1.0), 3), th.scale_tolerance,
                  abs(al.scale - 1.0) <= th.scale_tolerance, "Boyut hatasi"),
            Check("Buyuk kusur sayisi", len(big_defects), th.max_defects,
                  len(big_defects) <= th.max_defects, "Leke / cizik / lokal hata"),
        ]

        failed = [c for c in checks if not c.passed]
        ok = len(failed) == 0
        msg = "OK - Standarda uygun" if ok else "NOK: " + ", ".join(c.name for c in failed)

        overlay = _make_heatmap(aligned, heat)
        return InspectResult(
            ok=ok, found=True, message=msg, checks=checks,
            aligned=aligned, heatmap=overlay, defect_boxes=big_defects,
            frame_box=al.box, rotation_deg=al.rotation_deg, scale=al.scale,
            n_matches=al.n_matches,
        )


class AnomalyInspector:
    """GPU tabanli derin anomali denetimi (hibrit).

    Gorunum/kusur kontrolu derin anomali modeline (PatchCore) devredilir;
    hizalama + donme/olcek + renk kontrolleri klasik yontemle korunur.
    """

    def __init__(self, model: ReferenceModel, anomaly, thresholds: Thresholds):
        self.model = model
        self.anomaly = anomaly
        self.th = thresholds
        self.aligner = Aligner()

    def inspect(self, frame: np.ndarray) -> InspectResult:
        m = self.model
        th = self.th

        al = self.aligner.align(frame, m.kp, m.des, m.size, th.min_matches)
        if not al.ok:
            return InspectResult(ok=False, found=False, message=al.reason,
                                 n_matches=al.n_matches)
        aligned = al.aligned

        # --- Derin anomali (GPU) ---
        score, heat_small = self.anomaly.score(aligned)
        heat = cv2.resize(heat_small, (m.w, m.h))

        # --- Renk (CIEDE2000) ---
        cand_lab = _mean_lab(aligned, m.ink_mask)
        dE = float(deltaE_ciede2000(m.ref_lab[None, :], cand_lab[None, :])[0])

        # --- Anomali bolgeleri ---
        boxes = _heat_boxes(heat, th.anomaly_heat_level)
        roi_area = m.w * m.h
        big = [b for b in boxes if b[2] * b[3] >= th.defect_area_frac * roi_area]

        checks = [
            Check("Anomali skoru (GPU)", round(score, 3), th.anomaly_max,
                  score <= th.anomaly_max, "Ogrenilmis gorunumden sapma"),
            Check("Renk sapmasi (dE2000)", round(dE, 2), th.deltaE_max,
                  dE <= th.deltaE_max, "Ton / renk farki"),
            Check("Donme (derece)", round(abs(al.rotation_deg), 2), th.rotation_max_deg,
                  abs(al.rotation_deg) <= th.rotation_max_deg, "Aci kaymasi"),
            Check("Olcek sapmasi", round(abs(al.scale - 1.0), 3), th.scale_tolerance,
                  abs(al.scale - 1.0) <= th.scale_tolerance, "Boyut hatasi"),
            Check("Anomali bolgesi sayisi", len(big), th.max_defects,
                  len(big) <= th.max_defects, "Leke / cizik / lokal hata"),
        ]
        failed = [c for c in checks if not c.passed]
        ok = len(failed) == 0
        msg = "OK - Standarda uygun" if ok else "NOK: " + ", ".join(c.name for c in failed)

        overlay = _make_heatmap(aligned, (np.clip(heat, 0, 1) * 255).astype(np.uint8))
        return InspectResult(
            ok=ok, found=True, message=msg, checks=checks,
            aligned=aligned, heatmap=overlay, defect_boxes=big,
            frame_box=al.box, rotation_deg=al.rotation_deg, scale=al.scale,
            n_matches=al.n_matches,
        )


def build_inspector(model: ReferenceModel, thresholds: Thresholds, anomaly=None):
    """Profil tipine gore klasik veya derin denetleyici dondurur."""
    if anomaly is not None:
        return AnomalyInspector(model, anomaly, thresholds)
    return Inspector(model, thresholds)


# ---------------------------------------------------------------------------
# Yardimci fonksiyonlar
# ---------------------------------------------------------------------------
def _ink_mask(bgr: np.ndarray) -> np.ndarray:
    """Logonun (on plan) yaklasik maskesi. Arka plani kenar medyaninden tahmin eder."""
    gray = to_gray(bgr)
    border = np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ])
    bg = float(np.median(border))
    diff = cv2.absdiff(gray, np.full_like(gray, np.uint8(bg)))
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask.mean() < 5:  # neredeyse bos -> tum ROI
        mask = np.full_like(gray, 255)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.dilate(mask, k, iterations=1)
    return mask


def _mean_lab(bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = rgb2lab(rgb).reshape(-1, 3)
    if mask is not None:
        sel = mask.reshape(-1) > 0
        if sel.sum() > 10:
            return lab[sel].mean(axis=0)
    return lab.mean(axis=0)


def _edge_mask(ref_gray: np.ndarray) -> np.ndarray:
    """Referanstaki guclu kenarlar. Bu bolgelerde alt-piksel hizalama farki
    yanlis kusur uretir; bu yuzden kusur taramasinda bastirilir."""
    gx = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, em = cv2.threshold(mag, 40, 255, cv2.THRESH_BINARY)
    return cv2.dilate(em, np.ones((5, 5), np.uint8), iterations=1)


def _find_defects(diffmap: np.ndarray, ref_gray: np.ndarray, th: Thresholds):
    """Fark haritasindan kusur kutularini cikarir.

    - Referans kenarlari bastirilir (hizalama artefaktini onler).
    - Kenar seridi (ROI cercevesi) elenir.
    - Kalan fark bolgeleri = gercek leke / cizik / soluma / eksik baski.
    """
    d = np.clip(diffmap, 0, 1)
    d = cv2.GaussianBlur(d, (5, 5), 0)
    heat = (d * 255).astype(np.uint8)

    _, binmap = cv2.threshold(heat, int(th.defect_diff_level * 255), 255,
                              cv2.THRESH_BINARY)

    # Referans kenarlarini bastir
    binmap = cv2.bitwise_and(binmap, cv2.bitwise_not(_edge_mask(ref_gray)))

    # ROI cerceve seridini ele
    h, w = binmap.shape
    mb = max(3, int(0.04 * min(h, w)))
    safe = np.zeros_like(binmap)
    safe[mb:h - mb, mb:w - mb] = 255
    binmap = cv2.bitwise_and(binmap, safe)

    k = np.ones((3, 3), np.uint8)
    binmap = cv2.morphologyEx(binmap, cv2.MORPH_OPEN, k, iterations=1)
    binmap = cv2.morphologyEx(binmap, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(binmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    return boxes, heat


def _heat_boxes(heat: np.ndarray, level: float):
    """Anomali isi haritasindan (0..1) kusur kutulari."""
    h, w = heat.shape[:2]
    binmap = (heat >= level).astype(np.uint8) * 255
    mb = max(2, int(0.04 * min(h, w)))
    binmap[:mb, :] = 0
    binmap[-mb:, :] = 0
    binmap[:, :mb] = 0
    binmap[:, -mb:] = 0
    k = np.ones((3, 3), np.uint8)
    binmap = cv2.morphologyEx(binmap, cv2.MORPH_OPEN, k)
    binmap = cv2.morphologyEx(binmap, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(binmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours]


def _make_heatmap(bgr: np.ndarray, heat: np.ndarray) -> np.ndarray:
    cmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 0.65, cmap, 0.35, 0)
