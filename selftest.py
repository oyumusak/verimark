#!/usr/bin/env python3
"""Kamerasiz cekirdek testi.

Sentetik bir 'logo' uretir, referans olusturur, ardindan:
  - saglam bir kopya  -> OK beklenir
  - soluk baski       -> NOK beklenir
  - lekeli baski       -> NOK beklenir
  - kaymis/donmus     -> tolerans icinde OK / disinda NOK

UI'siz pipeline'in calistigini dogrular.
"""

import cv2
import numpy as np

from verimark.config import Thresholds
from verimark.inspector import Inspector, ReferenceModel
from verimark.profile import build_reference_image


def make_logo(w=300, h=200, faded=False, smudge=False):
    img = np.full((h, w, 3), 245, np.uint8)
    color = (180, 90, 40) if not faded else (220, 180, 160)
    cv2.circle(img, (w // 2, h // 2), 60, color, -1)
    cv2.putText(img, "VM", (w // 2 - 45, h // 2 + 20),
                cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 4)
    cv2.rectangle(img, (30, 30), (w - 30, h - 30), (60, 60, 60), 3)
    if smudge:
        cv2.circle(img, (80, 60), 18, (20, 20, 20), -1)
    return img


def put_on_scene(logo, dx=0, dy=0, angle=0.0):
    """Logoyu daha buyuk bir sahneye yerlestirir (ham kare simulasyonu)."""
    scene = np.full((400, 600, 3), 230, np.uint8)
    h, w = logo.shape[:2]
    if angle:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        logo = cv2.warpAffine(logo, M, (w, h), borderValue=(245, 245, 245))
    x, y = 150 + dx, 100 + dy
    scene[y:y + h, x:x + w] = logo
    return scene


def main():
    good = make_logo()
    samples = [put_on_scene(good, dx=i, dy=i) for i in range(0, 12, 2)]
    roi = (150, 100, 300, 200)

    ref_img = build_reference_image(samples, roi)
    model = ReferenceModel(ref_img)
    insp = Inspector(model, Thresholds())

    cases = {
        "saglam": put_on_scene(make_logo(), dx=3, dy=2),
        "soluk": put_on_scene(make_logo(faded=True)),
        "lekeli": put_on_scene(make_logo(smudge=True)),
        "donmus_15deg": put_on_scene(make_logo(), angle=15),
    }

    print(f"Referans murekkep kapsama: {model.ref_coverage:.3f}\n")
    for name, scene in cases.items():
        r = insp.inspect(scene)
        verdict = "OK " if r.ok else "NOK"
        fails = ", ".join(c.name for c in r.checks if not c.passed) if r.found else r.message
        print(f"[{verdict}] {name:14s} eslesme={r.n_matches:3d}  {fails}")


if __name__ == "__main__":
    main()
