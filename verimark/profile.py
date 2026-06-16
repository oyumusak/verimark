"""Urun profili: referans logo + esikler. Diske kaydet / yukle.

profiles/<isim>/
    reference.png   -> ortalama hizalanmis referans logo
    profile.json    -> meta + esikler
    samples/        -> ogretmede kullanilan ham numuneler (opsiyonel)
"""

from __future__ import annotations

import json
import os
import re
import time

import cv2
import numpy as np

from .alignment import Aligner
from .config import PROFILES_DIR, Thresholds
from .inspector import ReferenceModel


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return s or "profil"


def list_profiles() -> list[str]:
    out = []
    for d in sorted(os.listdir(PROFILES_DIR)):
        if os.path.isfile(os.path.join(PROFILES_DIR, d, "profile.json")):
            out.append(d)
    return out


def collect_aligned_crops(samples: list[np.ndarray],
                          roi: tuple[int, int, int, int]) -> list[np.ndarray]:
    """Tum numuneleri ilk numunenin ROI kirpilmasina hizalayip kirpma listesi dondurur.

    Ilk eleman referans kirpmasidir. Hizalanamayan numuneler atlanir.
    """
    x, y, w, h = roi
    base = samples[0][y:y + h, x:x + w].copy()
    aligner = Aligner()
    ref_kp, ref_des = aligner.detect(base)

    crops = [base]
    for s in samples[1:]:
        al = aligner.align(s, ref_kp, ref_des, (w, h), min_matches=10)
        if al.ok and al.aligned is not None:
            crops.append(al.aligned)
    return crops


def build_reference_image(samples: list[np.ndarray], roi: tuple[int, int, int, int]) -> np.ndarray:
    """Hizalanmis kirpmalarin MEDYANiyla temiz ama keskin bir referans uretir.

    Medyan, ortalamadan farkli olarak hafif hizalama hatasinda kenarlari
    bulanik yapmaz; bu da ORB'un referansta daha cok ozellik bulmasini saglar.
    """
    crops = collect_aligned_crops(samples, roi)
    stack = np.stack(crops, axis=0).astype(np.float32)
    return np.median(stack, axis=0).astype(np.uint8)


def save_profile(name: str, ref_bgr: np.ndarray, thresholds: Thresholds,
                 samples: list[np.ndarray] | None = None, meta: dict | None = None,
                 anomaly_model=None) -> str:
    folder = os.path.join(PROFILES_DIR, _safe_name(name))
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, "reference.png"), ref_bgr)

    mode = "anomaly" if anomaly_model is not None else "classic"
    if anomaly_model is not None:
        anomaly_model.save(os.path.join(folder, "anomaly.pt"))

    data = {
        "name": name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "thresholds": thresholds.asdict(),
        "meta": meta or {},
    }
    with open(os.path.join(folder, "profile.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if samples:
        sdir = os.path.join(folder, "samples")
        os.makedirs(sdir, exist_ok=True)
        for i, s in enumerate(samples):
            cv2.imwrite(os.path.join(sdir, f"sample_{i:02d}.png"), s)
    return folder


def load_profile(name: str) -> tuple[ReferenceModel, Thresholds, dict, object]:
    """(ReferenceModel, Thresholds, meta, anomaly_model|None) dondurur."""
    folder = os.path.join(PROFILES_DIR, _safe_name(name))
    ref = cv2.imread(os.path.join(folder, "reference.png"))
    if ref is None:
        raise FileNotFoundError(f"Referans goruntu bulunamadi: {folder}")
    with open(os.path.join(folder, "profile.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    th = Thresholds.fromdict(data.get("thresholds", {}))
    model = ReferenceModel(ref)

    anomaly_model = None
    apath = os.path.join(folder, "anomaly.pt")
    if data.get("mode") == "anomaly" and os.path.isfile(apath):
        from .anomaly import AnomalyModel, is_available
        if is_available():
            anomaly_model = AnomalyModel.load(apath)
    return model, th, data, anomaly_model


def delete_profile(name: str):
    import shutil
    folder = os.path.join(PROFILES_DIR, _safe_name(name))
    if os.path.isdir(folder):
        shutil.rmtree(folder)
