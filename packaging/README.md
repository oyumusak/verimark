# VeriMark — Windows .exe paketleme

Hedef: yazilim bilmeyen Windows kullanicisina tek bir `VeriMark.exe`
vermek (Python kurulumu yok, komut satiri yok).

Bu makine Linux oldugundan **Windows .exe'si yerelde uretilemez**
(PyInstaller capraz derleme yapmaz). Bu yuzden derleme **GitHub Actions**
uzerindeki Windows sunucusunda yapilir.

## .exe nasil alinir (Windows makineye gerek yok)

1. Projeyi GitHub'a push'la (asagidaki komutlar).
2. GitHub'da **Actions** sekmesini ac → "Windows EXE olustur" is akisi
   otomatik calisir (~5-10 dk).
3. Tamamlaninca calismaya tikla → en altta **Artifacts** altindaki
   `VeriMark-windows` dosyasini indir → icinden `VeriMark.exe` cikar.
4. Bu `VeriMark.exe` + `KULLANICI-KILAVUZU.txt` dosyalarini kullaniciya
   ver (USB / WeTransfer / mail).

### Surum yayinlamak (opsiyonel, daha duzenli)

Bir etiket (tag) push'larsan .exe ayrica **Releases** sayfasina eklenir:

```bash
git tag v1.0.0
git push origin v1.0.0
```

## Ilk push komutlari

```bash
cd /home/hiqermod/Desktop/yilportAll/verimark
git add -A
git commit -m "Windows .exe paketleme (PyInstaller + GitHub Actions)"
git push -u origin main
```

## Notlar

- **Klasik mod** paketlenir; `torch`/`torchvision` (GPU derin mod) bilerek
  haric tutulur — paket ~150-250 MB kalir, NVIDIA GPU gerektirmez.
  GPU surumu gerekirse `packaging/VeriMark.spec` icindeki `excludes`
  satiri kaldirilip `requirements-build.txt`'e CPU/CUDA torch eklenir
  (paket cok buyur).
- **Imzasiz exe**: Windows SmartScreen ilk acilista uyarir; kullanici
  "Yine de calistir" der. Kod imzalama sertifikasi (ucretli) gerekmez.
- **Ikon**: `packaging/VeriMark.ico` koyarsan otomatik kullanilir;
  yoksa varsayilan ikonla derlenir.
- Profiller calisma aninda `%LOCALAPPDATA%\VeriMark\profiles` altina
  yazilir (kurulum klasoru salt-okunur olabilir diye).
