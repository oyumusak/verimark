# VeriMark — Baskı/Logo Kalite Denetim Sistemi

Bir banttan geçen ürünlere basılan **logo/baskıyı**, telefonun IP kamera
uygulamasından canlı görüntüyle denetler. Önce sağlam baskıyı **öğretir**
(kalibrasyon), sonra her gelen baskıyı referansla karşılaştırıp **OK / NOK**
kararı verir, hatayı ekranda gösterir.

Yakaladığı kusur tipleri:

| Kusur | Yöntem |
|-------|--------|
| Soluk / eksik / fazla mürekkep | Mürekkep kapsama oranı + SSIM |
| Leke / kir / çizik / lokal hata | Fark haritası kusur blob'ları (ısı haritası) |
| Kayma / dönme / ölçek | Benzerlik dönüşümü (ORB + affine) çözümlemesi |
| Renk / ton farkı | Logo bölgesinde CIEDE2000 (ΔE) |

---

## Çalışma modları

| Mod | Yöntem | Donanım | Ne zaman |
|-----|--------|---------|----------|
| **Klasik** | Hizalama + SSIM + mürekkep + renk + fark blob'ları | CPU yeter | Hızlı, açıklanabilir; sabit baskıda |
| **Derin (GPU)** | PatchCore: önceden eğitilmiş CNN + bellek bankası | NVIDIA GPU | Logoyu "öğrenilmiş normal" olarak anlar; zor/çeşitli kusurlar |

Derin mod hizalama + dönme/ölçek + renk kontrollerini korur, **görünüm/kusur**
kısmını GPU'daki anomali modeline devreder (hibrit). Kalibrasyon ekranındaki
**"Derin öğrenme (GPU) ile öğren"** kutusuyla seçilir.

## Kurulum

Bu makinede hazır CUDA'lı ortam: `ayhanHocaSabahlama/yolo_env`
(torch 2.8 + cu128, GTX 1650). Doğrudan onunla çalıştır:

```bash
ENV=/home/hiqermod/Desktop/yilportAll/ayhanHocaSabahlama/yolo_env
$ENV/bin/python run.py
```

Sıfırdan kurulum (başka ortam için):

```bash
cd VeriMark
pip install -r requirements.txt
# GPU için torch'u CUDA wheel ile kur (requirements.txt içindeki nota bakın)
```

## Telefonu kamera yapma

1. Telefona bir IP kamera uygulaması kur:
   - **Android:** "IP Webcam" (önerilir) veya "DroidCam"
   - **iOS:** benzeri bir IP kamera uygulaması
2. Uygulamayı başlat → telefon ve bilgisayar **aynı Wi-Fi ağında** olsun.
3. Uygulamanın gösterdiği IP'yi not al (ör. `192.168.1.42`).
4. VeriMark'ta üstteki çubukta:
   - **Uygulama**: "IP Webcam (Android) - video"
   - **Telefon IP**: `192.168.1.42`
   - URL otomatik oluşur (`http://192.168.1.42:8080/video`) → **Bağlan**.

> Test için "Webcam (yerel 0)" seçip bilgisayarın kendi kamerasını da kullanabilirsin.

## Çalıştırma

```bash
python run.py
```

### 1) Öğret (Kalibrasyon) sekmesi
1. Kameraya bağlan.
2. Sağlam bir baskı görüntüsünde **fareyle logonun üzerine dikdörtgen çiz** (ROI).
3. Bant aktifken **"Sağlam Numune Yakala"** ile birkaç (≥5) sağlam örnek topla.
4. **"Referans Oluştur"** → ortalama referans logo üretilir (önizlemede görünür).
5. Profil adı ver → **"Profili Kaydet"**.

### 2) İzleme / Denetim sekmesi
- Bir profil yükle (Profiller sekmesinden) → otomatik aktif olur.
- **"Otomatik denetim"** kutusunu işaretle: her kare denetlenir.
- Büyük durum göstergesi **OK ✓ / NOK ✗ / LOGO YOK** olur.
- **"Kusur ısı haritası"** ile hatalı bölgeler renkli vurgulanır + kutu çizilir.
- Sağdaki listede her kontrolün ölçülen değeri ve limiti görünür.
- OK / NOK sayaçları tutulur.

### 3) Profiller / Ayarlar sekmesi
- Kayıtlı profilleri yükle / sil.
- **Genel Hassasiyet** kaydırıcısı: sol = toleranslı (az NOK), sağ = sıkı (çok NOK).
  Tüm eşikler bu tek kaydırıcıyla orantılı ayarlanır.

---

## Mimari

```
Kamera (IP) ─▶ Hizalama (ORB + affine) ─▶ Karşılaştırma ─▶ Karar ─▶ Ekran/Sayaç
                                          ├─ SSIM (yapı/soluma)
                                          ├─ Mürekkep kapsama
                                          ├─ Renk ΔE (CIEDE2000)
                                          ├─ Dönme / ölçek
                                          └─ Kusur blob'ları (ısı haritası)
```

| Dosya | Görev |
|-------|-------|
| `verimark/camera.py` | IP kamera / RTSP / webcam (thread'li taze kare okuma) |
| `verimark/alignment.py` | ORB özellik eşleme + benzerlik dönüşümü ile hizalama |
| `verimark/anomaly.py` | GPU derin anomali (PatchCore: CNN + bellek bankası) |
| `verimark/inspector.py` | Klasik + derin denetleyici, metrikler + OK/NOK kararı |
| `verimark/profile.py` | Ürün profili kaydet/yükle, referans üretme |
| `verimark/config.py` | Eşikler, kamera önayarları, hassasiyet ölçekleme |
| `verimark/ui/` | PyQt6 arayüz (3 sekme) + ROI seçim widget'ı |

## Doğrulama (kamerasız)

```bash
ENV=/home/hiqermod/Desktop/yilportAll/ayhanHocaSabahlama/yolo_env
$ENV/bin/python selftest.py        # klasik çekirdek (sağlam/soluk/lekeli/dönmüş)
$ENV/bin/python selftest_gpu.py    # GPU derin anomali (PatchCore)
$ENV/bin/python ui_smoketest.py    # UI'ı offscreen kurar: klasik + GPU akışı
```

## İpuçları / sınırlar

- **Aydınlatma** ne kadar sabitse o kadar iyi. Değişken ışıkta hassasiyeti
  düşürün veya kalibrasyonda farklı ışık koşullarından örnek toplayın.
- Logo görüntüde yeterince **dokulu/keskin** olmalı (ORB özellik bulabilsin).
  Çok küçük/bulanık görünüyorsa kamerayı yaklaştırın.
- **Derin (GPU) mod** logoyu öğrenilmiş normal görünüm olarak modeller; önceden
  tanımlanamayan leke/çizik/soluma gibi kusurlarda klasik yöntemden güçlüdür.
  GTX 1650 (4GB) için `resnet18` omurga varsayılan; daha güçlü GPU'da
  `AnomalyModel(backbone="wide_resnet50_2")` ile doğruluk artırılabilir.
- Derin modda kalibrasyon için **≥20 sağlam örnek** önerilir (ne kadar çok ve
  çeşitli, o kadar isabetli eşik). Az örnekte sistem augmentasyonla toleransı korur.
