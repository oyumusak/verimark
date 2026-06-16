#!/usr/bin/env python3
"""GPU derin anomali (PatchCore) testi.

Sagam orneklerle bellek bankasi olusturur, ardindan AnomalyInspector ile
sentetik kusur durumlarini sinar. Resnet18 agirliklari ilk calistirmada
internetten indirilir.
"""

import numpy as np

from verimark import anomaly as A
from verimark.config import Thresholds
from verimark.inspector import AnomalyInspector, ReferenceModel
from verimark.profile import build_reference_image, collect_aligned_crops
from selftest import make_logo, put_on_scene


def main():
    print("Cihaz:", A.device_info())
    if not A.is_available():
        print("torch yok; test atlandi.")
        return

    roi = (150, 100, 300, 200)
    samples = [put_on_scene(make_logo(), dx=i, dy=i) for i in range(0, 16, 2)]
    crops = collect_aligned_crops(samples, roi)

    model = A.AnomalyModel(input_size=256)
    model.fit(crops)
    print(f"Bellek bankasi: {model.memory.shape[0]} yama, esik={model.threshold:.3f}")

    ref_img = build_reference_image(samples, roi)
    ref_model = ReferenceModel(ref_img)
    insp = AnomalyInspector(ref_model, model, Thresholds())

    cases = {
        "saglam": put_on_scene(make_logo(), dx=3, dy=2),
        "soluk": put_on_scene(make_logo(faded=True)),
        "lekeli": put_on_scene(make_logo(smudge=True)),
        "donmus_15deg": put_on_scene(make_logo(), angle=15),
    }
    print()
    for name, scene in cases.items():
        r = insp.inspect(scene)
        verdict = "OK " if r.ok else "NOK"
        ascore = next((c.value for c in r.checks if "Anomali skoru" in c.name), "-")
        fails = ", ".join(c.name for c in r.checks if not c.passed) if r.found else r.message
        print(f"[{verdict}] {name:14s} anomali={ascore}  {fails}")


if __name__ == "__main__":
    main()
