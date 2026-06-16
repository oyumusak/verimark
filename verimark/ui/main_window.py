"""VeriMark ana penceresi."""

from __future__ import annotations

import sys

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QComboBox, QHBoxLayout, QVBoxLayout, QGridLayout, QTabWidget, QGroupBox,
    QListWidget, QMessageBox, QSlider, QCheckBox,
)

from .. import __appname__, __version__
from .. import anomaly as anomaly_mod
from ..camera import IPCamera
from ..config import CAMERA_PRESETS, DEFAULT_IP, Thresholds
from ..inspector import build_inspector
from ..profile import (
    build_reference_image, collect_aligned_crops, save_profile, load_profile,
    list_profiles, delete_profile,
)
from .image_view import ImageView


def _status_box(color: str) -> str:
    return (f"background:{color}; color:white; border-radius:8px; "
            f"padding:10px; font-size:22px; font-weight:bold;")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{__appname__} - Baski/Logo Kalite Denetimi  v{__version__}")
        self.resize(1180, 760)

        self.cam: IPCamera | None = None
        self.inspector = None
        self.thresholds = Thresholds()
        self.samples: list[np.ndarray] = []
        self.current_roi = None
        self.auto_inspect = False
        self.ok_count = 0
        self.nok_count = 0
        self._stable_state = None
        self._stable_frames = 0
        self.active_profile = "-"
        self.active_model = None
        self.active_anomaly = None
        self._anomaly_model = None  # kalibrasyonda egitilen gecici model

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)  # ~30 fps

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.addWidget(self._build_conn_bar())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_inspect_tab(), "  İzleme / Denetim  ")
        self.tabs.addTab(self._build_learn_tab(), "  Öğret (Kalibrasyon)  ")
        self.tabs.addTab(self._build_profiles_tab(), "  Profiller / Ayarlar  ")
        root.addWidget(self.tabs)

    def _build_conn_bar(self) -> QWidget:
        box = QGroupBox("Kamera Baglantisi (Telefon IP Kamera)")
        lay = QHBoxLayout(box)

        self.preset = QComboBox()
        self.preset.addItems(CAMERA_PRESETS.keys())
        self.preset.currentTextChanged.connect(self._update_url)

        self.ip_edit = QLineEdit(DEFAULT_IP)
        self.ip_edit.setFixedWidth(140)
        self.ip_edit.textChanged.connect(self._update_url)

        self.url_edit = QLineEdit()
        self.url_edit.setMinimumWidth(280)

        self.btn_connect = QPushButton("Baglan")
        self.btn_connect.clicked.connect(self._toggle_connect)

        self.conn_light = QLabel("●  Bagli degil")
        self.conn_light.setStyleSheet("color:#cc4444; font-weight:bold;")

        lay.addWidget(QLabel("Uygulama:"))
        lay.addWidget(self.preset)
        lay.addWidget(QLabel("Telefon IP:"))
        lay.addWidget(self.ip_edit)
        lay.addWidget(QLabel("URL:"))
        lay.addWidget(self.url_edit, 1)
        lay.addWidget(self.btn_connect)
        lay.addWidget(self.conn_light)
        self._update_url()
        return box

    def _build_inspect_tab(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)

        self.view_inspect = ImageView()
        lay.addWidget(self.view_inspect, 3)

        side = QVBoxLayout()
        self.lbl_status = QLabel("HAZIR")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet(_status_box("#555"))
        side.addWidget(self.lbl_status)

        self.lbl_profile = QLabel("Aktif profil: -")
        self.lbl_profile.setStyleSheet("font-weight:bold;")
        side.addWidget(self.lbl_profile)

        self.chk_auto = QCheckBox("Otomatik denetim (surekli)")
        self.chk_auto.stateChanged.connect(
            lambda s: setattr(self, "auto_inspect", bool(s)))
        side.addWidget(self.chk_auto)

        self.chk_heat = QCheckBox("Kusur isi haritasini goster")
        side.addWidget(self.chk_heat)

        self.btn_inspect_once = QPushButton("Bu Kareyi Denetle")
        self.btn_inspect_once.clicked.connect(self._inspect_once)
        side.addWidget(self.btn_inspect_once)

        cnt = QHBoxLayout()
        self.lbl_ok = QLabel("OK: 0")
        self.lbl_ok.setStyleSheet("color:#2a9d4a; font-weight:bold; font-size:16px;")
        self.lbl_nok = QLabel("NOK: 0")
        self.lbl_nok.setStyleSheet("color:#c0392b; font-weight:bold; font-size:16px;")
        cnt.addWidget(self.lbl_ok)
        cnt.addWidget(self.lbl_nok)
        btn_reset = QPushButton("Sifirla")
        btn_reset.clicked.connect(self._reset_counts)
        cnt.addWidget(btn_reset)
        side.addLayout(cnt)

        gb = QGroupBox("Denetim Detaylari")
        gl = QVBoxLayout(gb)
        self.checks_list = QListWidget()
        gl.addWidget(self.checks_list)
        side.addWidget(gb, 1)

        lay.addLayout(side, 2)
        return w

    def _build_learn_tab(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)

        self.view_learn = ImageView(selectable=True)
        self.view_learn.roiSelected.connect(self._on_roi)
        lay.addWidget(self.view_learn, 3)

        side = QVBoxLayout()
        info = QLabel(
            "1) Kameraya baglanin.\n"
            "2) SAGLAM bir baski goruntusunde, fareyle logonun uzerine\n"
            "    dikdortgen cizerek bolgeyi (ROI) secin.\n"
            "3) Bant aktifken birkac saglam ornek yakalayin (>= 5 onerilir).\n"
            "4) 'Referans Olustur' ile ortalama referansi uretin.\n"
            "5) Profil adi verip kaydedin.")
        info.setStyleSheet("color:#bbb;")
        side.addWidget(info)

        self.lbl_roi = QLabel("ROI: secilmedi")
        side.addWidget(self.lbl_roi)

        self.btn_capture = QPushButton("Saglam Numune Yakala")
        self.btn_capture.clicked.connect(self._capture_sample)
        side.addWidget(self.btn_capture)

        self.lbl_samples = QLabel("Numune: 0")
        side.addWidget(self.lbl_samples)

        btn_clear = QPushButton("Numuneleri Temizle")
        btn_clear.clicked.connect(self._clear_samples)
        side.addWidget(btn_clear)

        self.chk_gpu = QCheckBox("Derin ogrenme (GPU) ile ogren")
        gpu_ok = anomaly_mod.is_available()
        self.chk_gpu.setChecked(gpu_ok)
        self.chk_gpu.setEnabled(gpu_ok)
        side.addWidget(self.chk_gpu)
        self.lbl_device = QLabel(anomaly_mod.device_info())
        self.lbl_device.setStyleSheet("color:#7aa;")
        side.addWidget(self.lbl_device)

        self.btn_build = QPushButton("Referans Olustur (onizleme)")
        self.btn_build.clicked.connect(self._build_reference)
        side.addWidget(self.btn_build)

        self.view_ref = ImageView()
        self.view_ref.setMinimumSize(240, 180)
        side.addWidget(QLabel("Referans onizleme:"))
        side.addWidget(self.view_ref)

        row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Profil adi (orn: urun_A_logo)")
        row.addWidget(self.name_edit)
        self.btn_save = QPushButton("Profili Kaydet")
        self.btn_save.clicked.connect(self._save_profile)
        row.addWidget(self.btn_save)
        side.addLayout(row)

        lay.addLayout(side, 2)
        self._ref_preview = None
        return w

    def _build_profiles_tab(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)

        left = QVBoxLayout()
        left.addWidget(QLabel("Kayitli Profiller:"))
        self.profile_list = QListWidget()
        self.profile_list.itemDoubleClicked.connect(lambda _: self._load_selected())
        left.addWidget(self.profile_list)
        b1 = QPushButton("Yukle / Aktif Et")
        b1.clicked.connect(self._load_selected)
        left.addWidget(b1)
        b2 = QPushButton("Sil")
        b2.clicked.connect(self._delete_selected)
        left.addWidget(b2)
        b3 = QPushButton("Listeyi Yenile")
        b3.clicked.connect(self._refresh_profiles)
        left.addWidget(b3)
        lay.addLayout(left, 1)

        gb = QGroupBox("Esikler / Hassasiyet")
        gl = QGridLayout(gb)

        gl.addWidget(QLabel("Genel Hassasiyet"), 0, 0)
        self.sld_sens = QSlider(Qt.Orientation.Horizontal)
        self.sld_sens.setRange(0, 100)
        self.sld_sens.setValue(50)
        self.sld_sens.valueChanged.connect(self._apply_sensitivity)
        gl.addWidget(self.sld_sens, 0, 1)
        self.lbl_sens = QLabel("50 (normal)")
        gl.addWidget(self.lbl_sens, 0, 2)

        note = QLabel("Sol = toleransli (daha az NOK)   |   Sag = siki (daha cok NOK)")
        note.setStyleSheet("color:#999;")
        gl.addWidget(note, 1, 0, 1, 3)

        self.th_labels = QLabel()
        self.th_labels.setStyleSheet("font-family:monospace;")
        gl.addWidget(self.th_labels, 2, 0, 1, 3)
        self._refresh_threshold_labels()

        lay.addWidget(gb, 2)
        self._refresh_profiles()
        return w

    # ------------------------------------------------------------- kamera
    def _update_url(self):
        tpl = CAMERA_PRESETS[self.preset.currentText()]
        self.url_edit.setText(tpl.replace("{ip}", self.ip_edit.text().strip()))

    def _toggle_connect(self):
        if self.cam is not None:
            self.cam.release()
            self.cam = None
            self.btn_connect.setText("Baglan")
            self.conn_light.setText("●  Bagli degil")
            self.conn_light.setStyleSheet("color:#cc4444; font-weight:bold;")
            return

        url = self.url_edit.text().strip()
        self.conn_light.setText("●  Baglaniyor...")
        self.conn_light.setStyleSheet("color:#e0a800; font-weight:bold;")
        QApplication.processEvents()
        cam = IPCamera(url)
        if cam.open():
            self.cam = cam
            self.btn_connect.setText("Baglantiyi Kes")
            self.conn_light.setText("●  Bagli")
            self.conn_light.setStyleSheet("color:#2a9d4a; font-weight:bold;")
        else:
            QMessageBox.warning(self, "Baglanti Hatasi", cam.error)
            self.conn_light.setText("●  Bagli degil")
            self.conn_light.setStyleSheet("color:#cc4444; font-weight:bold;")

    def _tick(self):
        if self.cam is None:
            return
        frame = self.cam.read()
        if frame is None:
            return

        idx = self.tabs.currentIndex()
        if idx == 1:
            self.view_learn.set_frame(frame)
        elif idx == 0:
            self._update_inspect_view(frame)

    # ------------------------------------------------------------ denetim
    def _update_inspect_view(self, frame: np.ndarray):
        if self.inspector is None or not self.auto_inspect:
            self.view_inspect.set_frame(frame)
            if self.inspector is None:
                self.lbl_status.setText("PROFIL YOK")
                self.lbl_status.setStyleSheet(_status_box("#777"))
            return
        res = self.inspector.inspect(frame)
        self._render_result(frame, res)
        self._update_counts(res)

    def _inspect_once(self):
        if self.cam is None or self.inspector is None:
            QMessageBox.information(self, "Bilgi",
                                    "Once kameraya baglanin ve bir profil yukleyin.")
            return
        frame = self.cam.read()
        if frame is None:
            return
        res = self.inspector.inspect(frame)
        self._render_result(frame, res)

    def _render_result(self, frame: np.ndarray, res):
        disp = frame.copy()
        if res.found and res.frame_box is not None:
            pts = res.frame_box.astype(int).reshape(-1, 2)
            color = (0, 200, 0) if res.ok else (0, 0, 255)
            cv2.polylines(disp, [pts], True, color, 3)

        if self.chk_heat.isChecked() and res.heatmap is not None and res.found:
            h, w = res.heatmap.shape[:2]
            disp[0:h, 0:w] = cv2.resize(res.heatmap, (w, h))
            for (x, y, bw, bh) in res.defect_boxes:
                cv2.rectangle(disp, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

        self.view_inspect.set_frame(disp)

        if not res.found:
            self.lbl_status.setText("LOGO YOK")
            self.lbl_status.setStyleSheet(_status_box("#c0392b"))
        elif res.ok:
            self.lbl_status.setText("OK ✓")
            self.lbl_status.setStyleSheet(_status_box("#2a9d4a"))
        else:
            self.lbl_status.setText("NOK ✗")
            self.lbl_status.setStyleSheet(_status_box("#c0392b"))

        self.checks_list.clear()
        self.checks_list.addItem(f"Durum: {res.message}")
        self.checks_list.addItem(f"Eslesme: {res.n_matches}")
        for c in res.checks:
            mark = "✓" if c.passed else "✗"
            self.checks_list.addItem(
                f"{mark} {c.name}: {c.value}  (limit {c.limit})  - {c.detail}")

    def _update_counts(self, res):
        state = "ok" if (res.found and res.ok) else "nok"
        if state == self._stable_state:
            self._stable_frames += 1
        else:
            self._stable_state = state
            self._stable_frames = 1
            self._counted = False
        # ayni durum 6 kare stabil olunca bir kez say (debounce)
        if self._stable_frames == 6:
            if state == "ok":
                self.ok_count += 1
            else:
                self.nok_count += 1
            self.lbl_ok.setText(f"OK: {self.ok_count}")
            self.lbl_nok.setText(f"NOK: {self.nok_count}")

    def _reset_counts(self):
        self.ok_count = self.nok_count = 0
        self.lbl_ok.setText("OK: 0")
        self.lbl_nok.setText("NOK: 0")

    # ----------------------------------------------------------- ogretme
    def _on_roi(self, roi):
        self.current_roi = roi
        self.lbl_roi.setText(f"ROI: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")

    def _capture_sample(self):
        if self.cam is None:
            QMessageBox.information(self, "Bilgi", "Once kameraya baglanin.")
            return
        frame = self.cam.read()
        if frame is None:
            return
        self.samples.append(frame.copy())
        self.lbl_samples.setText(f"Numune: {len(self.samples)}")

    def _clear_samples(self):
        self.samples.clear()
        self.lbl_samples.setText("Numune: 0")

    def _build_reference(self):
        if not self.samples:
            QMessageBox.information(self, "Bilgi", "Once en az 1 numune yakalayin.")
            return
        if self.current_roi is None:
            QMessageBox.information(self, "Bilgi",
                                    "Once goruntude logo bolgesini (ROI) secin.")
            return
        try:
            crops = collect_aligned_crops(self.samples, self.current_roi)
            ref = build_reference_image(self.samples, self.current_roi)
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Referans olusturulamadi:\n{e}")
            return
        self._ref_preview = ref
        self.view_ref.set_frame(ref)

        # Derin model egitimi (istege bagli, GPU)
        self._anomaly_model = None
        if self.chk_gpu.isChecked() and anomaly_mod.is_available():
            try:
                self.lbl_device.setText("Derin model egitiliyor...")
                QApplication.processEvents()
                model = anomaly_mod.AnomalyModel()
                model.fit(crops)
                self._anomaly_model = model
                self.lbl_device.setText(
                    f"{anomaly_mod.device_info()}  |  banka={model.memory.shape[0]} "
                    f"yama, esik={model.threshold:.2f}")
            except Exception as e:
                QMessageBox.warning(self, "GPU Hata",
                                    f"Derin model egitilemedi:\n{e}")
                self.lbl_device.setText(anomaly_mod.device_info())

    def _save_profile(self):
        if self._ref_preview is None:
            QMessageBox.information(self, "Bilgi", "Once 'Referans Olustur'a basin.")
            return
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.information(self, "Bilgi", "Profil adi girin.")
            return
        folder = save_profile(name, self._ref_preview, self.thresholds,
                              samples=self.samples, anomaly_model=self._anomaly_model)
        mode = "DERIN (GPU)" if self._anomaly_model is not None else "KLASIK"
        QMessageBox.information(self, "Kaydedildi",
                               f"Profil kaydedildi ({mode}):\n{folder}")
        self._refresh_profiles()
        self._activate_profile(name)

    # ---------------------------------------------------------- profiller
    def _refresh_profiles(self):
        self.profile_list.clear()
        self.profile_list.addItems(list_profiles())

    def _selected_profile(self):
        it = self.profile_list.currentItem()
        return it.text() if it else None

    def _load_selected(self):
        name = self._selected_profile()
        if not name:
            return
        self._activate_profile(name)
        QMessageBox.information(self, "Yuklendi", f"Aktif profil: {name}")

    def _activate_profile(self, name: str):
        try:
            model, th, data, anomaly = load_profile(name)
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"Profil yuklenemedi:\n{e}")
            return
        self.thresholds = th
        self.active_profile = name
        self.active_model = model
        self.active_anomaly = anomaly
        self._apply_sensitivity(self.sld_sens.value())
        mode = "DERIN (GPU)" if anomaly is not None else "KLASIK"
        self.lbl_profile.setText(f"Aktif profil: {name}  [{mode}]")
        self.tabs.setCurrentIndex(0)

    def _delete_selected(self):
        name = self._selected_profile()
        if not name:
            return
        if QMessageBox.question(self, "Onay", f"'{name}' silinsin mi?") == \
                QMessageBox.StandardButton.Yes:
            delete_profile(name)
            self._refresh_profiles()

    # --------------------------------------------------------- esikler
    def _apply_sensitivity(self, val: int):
        names = {0: "cok toleransli", 25: "toleransli", 50: "normal",
                 75: "siki", 100: "cok siki"}
        closest = min(names, key=lambda k: abs(k - val))
        self.lbl_sens.setText(f"{val} ({names[closest]})")
        scaled = self.thresholds.scaled(val)
        if self.active_model is not None:
            self.inspector = build_inspector(self.active_model, scaled,
                                             self.active_anomaly)
        self._refresh_threshold_labels(scaled)

    def _refresh_threshold_labels(self, th: Thresholds | None = None):
        th = th or self.thresholds
        deep = self.active_anomaly is not None
        self.th_labels.setText(
            (f"Anomali skoru <= {th.anomaly_max:.2f}  (DERIN/GPU)\n"
             if deep else f"SSIM >= {th.ssim_min:.2f}\n"
                          f"Murekkep tol. <= {th.ink_tolerance:.2f}\n"
                          f"Kusur fark esigi <= {th.defect_diff_level:.2f}\n") +
            f"Renk dE <= {th.deltaE_max:.1f}\n"
            f"Donme <= {th.rotation_max_deg:.1f} derece\n"
            f"Olcek tol. <= {th.scale_tolerance:.2f}")

    def closeEvent(self, e):
        if self.cam is not None:
            self.cam.release()
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Sans", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
