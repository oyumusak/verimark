#!/usr/bin/env python3
"""UI'i offscreen surup TUM akisi test eder: ogret -> (GPU) egit -> kaydet
-> aktive et -> denetle. Klasik ve derin (GPU) modlarinin ikisini de dener."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from verimark.ui.main_window import MainWindow
from verimark import anomaly as A
from verimark.profile import delete_profile
from selftest import make_logo, put_on_scene


def run_flow(win, use_gpu, profile_name):
    win.samples = [put_on_scene(make_logo(), dx=i, dy=i) for i in range(0, 16, 2)]
    win.current_roi = (150, 100, 300, 200)
    win.chk_gpu.setChecked(use_gpu)
    win._build_reference()
    win.name_edit.setText(profile_name)
    win._save_profile()       # kaydeder + aktive eder
    win.auto_inspect = True

    scene = put_on_scene(make_logo(smudge=True))
    win._update_inspect_view(scene)
    status = win.lbl_status.text()
    prof = win.lbl_profile.text()
    deep = win.active_anomaly is not None
    print(f"[{'GPU ' if use_gpu else 'KLAS'}] status={status!r}  deep={deep}  {prof}")
    assert "NOK" in status, f"{profile_name}: NOK bekleniyordu"
    if use_gpu:
        assert deep, "Derin model aktif olmaliydi"


def main():
    app = QApplication([])
    win = MainWindow()
    print("Cihaz:", A.device_info())

    try:
        run_flow(win, use_gpu=False, profile_name="_test_classic")
        if A.is_available():
            run_flow(win, use_gpu=True, profile_name="_test_gpu")
    finally:
        for n in ("_test_classic", "_test_gpu"):
            try:
                delete_profile(n)
            except Exception:
                pass
        win.close()
    print("\nUI SMOKETEST OK")


if __name__ == "__main__":
    main()
