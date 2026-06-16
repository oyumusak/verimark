# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — VeriMark tek dosya (.exe) klasik mod.

GPU/derin mod (torch, torchvision) bilerek HARIC tutulur: hedef makinede
NVIDIA GPU yoksa gereksiz ve paketi ~2-3 GB buyutur. Klasik mod tum kusur
tiplerini yakalar.

Windows uzerinde uretilir (GitHub Actions / Windows runner):
    pyinstaller packaging/VeriMark.spec
Cikti: dist/VeriMark.exe
"""

import os

from PyInstaller.utils.hooks import collect_submodules

# scikit-image bazi alt modulleri tembel (lazy) yukler; PyInstaller bunlari
# kacirabilir. SSIM (skimage.metrics) ve CIEDE2000 (skimage.color) gerekli.
hiddenimports = collect_submodules("skimage")

# Opsiyonel ikon: packaging/VeriMark.ico varsa kullanilir.
_icon_path = os.path.join(SPECPATH, "VeriMark.ico")
_icon = _icon_path if os.path.exists(_icon_path) else None

# torch'u kesinlikle paketleme (ortamda kurulu olsa bile).
excludes = ["torch", "torchvision", "torchaudio"]


a = Analysis(
    ["..\\run.py"],
    pathex=[".."],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="VeriMark",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,           # GUI uygulamasi: siyah konsol penceresi acilmaz
    disable_windowed_traceback=False,
    icon=_icon,
)
