"""Ozellik tabanli hizalama (ORB + homography).

Gelen karedeki logoyu referans logonun koordinat sistemine oturtur.
Kayma / donme / olcek farklarini tolere eder ve bunlari olcer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


@dataclass
class AlignResult:
    ok: bool
    n_matches: int
    aligned: np.ndarray | None      # referans boyutuna oturtulmus aday goruntu (BGR)
    homography: np.ndarray | None
    rotation_deg: float
    scale: float                    # 1.0 = referansla ayni olcek
    box: np.ndarray | None          # logonun ham karedeki konumu (4 nokta)
    reason: str = ""


class Aligner:
    def __init__(self, nfeatures: int = 3000):
        self.orb = cv2.ORB_create(nfeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        # Dusuk kontrastli baski (orn. peceteye soluk logo) icin yerel kontrast
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, img: np.ndarray):
        gray = self._clahe.apply(to_gray(img))
        return self.orb.detectAndCompute(gray, None)

    def align(self, frame: np.ndarray, ref_kp, ref_des,
              ref_size: tuple[int, int], min_matches: int) -> AlignResult:
        """frame icindeki logoyu ref koordinatlarina warp eder.

        ref_size = (genislik, yukseklik)
        """
        ref_w, ref_h = ref_size

        if ref_des is None or len(ref_kp) < 2:
            return AlignResult(False, 0, None, None, 0.0, 1.0, None,
                               "Referans ozellikleri yetersiz.")

        kp, des = self.detect(frame)
        if des is None or len(kp) < 2:
            return AlignResult(False, 0, None, None, 0.0, 1.0, None,
                               "Karede ozellik bulunamadi.")

        # knn + Lowe oran testi
        try:
            knn = self.matcher.knnMatch(ref_des, des, k=2)
        except cv2.error:
            return AlignResult(False, 0, None, None, 0.0, 1.0, None,
                               "Eslestirme hatasi.")

        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.82 * n.distance:
                good.append(m)

        if len(good) < min_matches:
            return AlignResult(False, len(good), None, None, 0.0, 1.0, None,
                               "Logo bulunamadi / eksik baski (yetersiz eslesme).")

        src = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Konveyor baskisi duzlemseldir: oteleme + donme + olcek (benzerlik donusumu).
        # M: referans -> kare
        M_r2f, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if M_r2f is None:
            return AlignResult(False, len(good), None, None, 0.0, 1.0, None,
                               "Donusum hesaplanamadi.")

        inliers = int(mask.sum()) if mask is not None else len(good)
        if inliers < min_matches:
            return AlignResult(False, inliers, None, None, 0.0, 1.0, None,
                               "Tutarli eslesme yetersiz (gurultu).")

        # Logonun karedeki kosegen kutusu
        ref_corners = np.float32([[0, 0], [ref_w, 0],
                                  [ref_w, ref_h], [0, ref_h]]).reshape(-1, 1, 2)
        box = cv2.transform(ref_corners, M_r2f)

        # kare -> referans  (aday goruntuyu referans cercevesine oturt)
        M_f2r = cv2.invertAffineTransform(M_r2f)
        aligned = cv2.warpAffine(frame, M_f2r, (ref_w, ref_h))

        rot, scale = _decompose(M_r2f)
        return AlignResult(True, inliers, aligned, M_f2r, rot, scale, box)


def _decompose(M: np.ndarray) -> tuple[float, float]:
    """Benzerlik donusumunden (2x3) donme (derece) ve olcek."""
    a, c = M[0, 0], M[1, 0]
    scale = math.hypot(a, c)
    rot = math.degrees(math.atan2(c, a))
    if rot > 90:
        rot -= 180
    elif rot < -90:
        rot += 180
    return rot, scale
