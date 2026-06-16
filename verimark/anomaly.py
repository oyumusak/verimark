"""GPU tabanli derin anomali tespiti (kendi icinde PatchCore).

Onceden egitilmis bir CNN (torchvision) ile orta seviye ozellikler cikarilir,
SAGLAM orneklerden bir 'bellek bankasi' (memory bank) olusturulur. Denetimde
her yamanin (patch) bankaya en yakin mesafesi anomali skorunu/haritasini verir.

- Sadece SAGLAM orneklerle ogrenir (kusur etiketi gerekmez).
- GPU varsa otomatik kullanir (torch.cuda), yoksa CPU.
- Anomalib bagimliligi YOK; sadece torch + torchvision.

torch yoksa bu modul import edilebilir ama AnomalyModel olusturulamaz
(is_available() False doner) — uygulama klasik modda calismaya devam eder.
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torchvision.models import (
        resnet18, ResNet18_Weights,
        wide_resnet50_2, Wide_ResNet50_2_Weights,
    )
    _TORCH_OK = True
except Exception:  # pragma: no cover
    _TORCH_OK = False


def is_available() -> bool:
    return _TORCH_OK


def device_info() -> str:
    if not _TORCH_OK:
        return "torch yok (klasik mod)"
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return "CPU (GPU bulunamadi)"


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class AnomalyModel:
    """PatchCore tarzi anomali modeli."""

    def __init__(self, input_size: int = 256, backbone: str = "resnet18",
                 max_bank: int = 8000, device: str | None = None):
        if not _TORCH_OK:
            raise RuntimeError("torch/torchvision kurulu degil.")
        self.input_size = int(input_size)
        self.backbone_name = backbone
        self.max_bank = int(max_bank)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.memory: "torch.Tensor | None" = None  # (N, C)
        self.fmap_hw: tuple[int, int] = (0, 0)
        self.threshold: float = 1.0     # ham skor esigi (kalibre edilir)
        self.train_mean: float = 0.0
        self.train_std: float = 0.0

        self._feats: dict[str, "torch.Tensor"] = {}
        self._build_backbone()

    # ------------------------------------------------------------- backbone
    def _build_backbone(self):
        if self.backbone_name == "wide_resnet50_2":
            net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        else:
            self.backbone_name = "resnet18"
            net = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.layers = ["layer2", "layer3"]  # orta seviye dokular
        net.eval().to(self.device)
        for p in net.parameters():
            p.requires_grad_(False)
        for name in self.layers:
            getattr(net, name).register_forward_hook(self._make_hook(name))
        self.net = net
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def _make_hook(self, name):
        def hook(_m, _i, out):
            self._feats[name] = out
        return hook

    # ------------------------------------------------------------- ozellik
    def _preprocess(self, bgr: np.ndarray) -> "torch.Tensor":
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size))
        t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        t = t.to(self.device)
        return (t - self._mean) / self._std

    @torch.no_grad()
    def _embed(self, bgr: np.ndarray) -> "torch.Tensor":
        """(C, h, w) ozellik haritasi."""
        x = self._preprocess(bgr)
        self._feats.clear()
        self.net(x)
        target = self._feats[self.layers[0]].shape[-2:]
        feats = []
        for name in self.layers:
            f = self._feats[name]
            if f.shape[-2:] != target:
                f = F.interpolate(f, size=target, mode="bilinear", align_corners=False)
            feats.append(f)
        emb = torch.cat(feats, dim=1)[0]  # (C, h, w)
        self.fmap_hw = (int(target[0]), int(target[1]))
        return emb

    def _patches(self, emb: "torch.Tensor") -> "torch.Tensor":
        c = emb.shape[0]
        return emb.reshape(c, -1).t().contiguous()  # (h*w, C)

    @torch.no_grad()
    def _min_dist(self, vecs: "torch.Tensor") -> "torch.Tensor":
        """Her yamanin bankaya en yakin L2 mesafesi (parcali, bellek dostu)."""
        out = torch.empty(vecs.shape[0], device=vecs.device)
        chunk = 2048
        for i in range(0, vecs.shape[0], chunk):
            d = torch.cdist(vecs[i:i + chunk], self.memory)  # (chunk, N)
            out[i:i + chunk] = d.min(dim=1).values
        return out

    # ------------------------------------------------------------- egitim
    @staticmethod
    def _augment(bgr: np.ndarray) -> list[np.ndarray]:
        """Iyi-huylu varyasyonlar: hafif bulaniklik, parlaklik, 1px kayma.

        Normal manifoldunu genisletir; benzer ama kusursuz baskilar OK kalir.
        """
        h, w = bgr.shape[:2]
        out = [bgr]
        out.append(cv2.GaussianBlur(bgr, (3, 3), 0))
        out.append(np.clip(bgr.astype(np.int16) * 0.92, 0, 255).astype(np.uint8))
        out.append(np.clip(bgr.astype(np.int16) * 1.08, 0, 255).astype(np.uint8))
        for dx, dy in ((1, 0), (0, 1), (-1, -1)):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            out.append(cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT))
        return out

    @torch.no_grad()
    def fit(self, samples: list[np.ndarray]):
        """Hizalanmis SAGLAM logo kirpmalariyla bellek bankasini olusturur."""
        if not samples:
            raise ValueError("En az 1 saglam ornek gerekli.")

        # Her saglam ornek + iyi-huylu varyasyonlari (augmentasyon)
        aug_imgs = []
        for s in samples:
            aug_imgs.extend(self._augment(s))

        banks = [self._patches(self._embed(v)) for v in aug_imgs]
        bank = torch.cat(banks, dim=0)
        if bank.shape[0] > self.max_bank:  # rastgele alt-ornekleme (coreset yerine)
            idx = torch.randperm(bank.shape[0], device=bank.device)[:self.max_bank]
            bank = bank[idx]
        self.memory = bank.contiguous()

        # Esik: iyi-huylu varyasyon skorlarinin dagilimindan
        scores = np.array([self._raw_score(v) for v in aug_imgs], dtype=np.float64)
        self.train_mean = float(scores.mean())
        self.train_std = float(scores.std() + 1e-6)
        self.threshold = float(max(scores.max() * 1.30,
                                   self.train_mean + 5.0 * self.train_std))
        return self

    @torch.no_grad()
    def _raw_score(self, bgr: np.ndarray) -> float:
        emb = self._embed(bgr)
        dmin = self._min_dist(self._patches(emb))
        # PatchCore: en yuksek yama mesafesi (uc deger), gurultuye karsi top-k ort.
        k = max(1, int(0.002 * dmin.numel()))
        topk = torch.topk(dmin, k).values
        return float(topk.mean().item())

    # ------------------------------------------------------------- skor
    @torch.no_grad()
    def score(self, bgr: np.ndarray):
        """(normalize_skor, anomali_haritasi 0..1) dondurur.

        normalize_skor: 1.0 = kalibre esik. >1.0 => anomali (NOK).
        """
        emb = self._embed(bgr)
        h, w = self.fmap_hw
        dmin = self._min_dist(self._patches(emb))
        amap = dmin.reshape(h, w)

        k = max(1, int(0.002 * dmin.numel()))
        raw = float(torch.topk(dmin, k).values.mean().item())
        norm = raw / max(1e-6, self.threshold)

        heat = (amap / max(1e-6, self.threshold)).clamp(0, 2.0) / 2.0
        heat = heat.detach().cpu().numpy().astype(np.float32)
        return norm, heat

    # ------------------------------------------------------------- kayit
    def state(self) -> dict:
        return {
            "input_size": self.input_size,
            "backbone": self.backbone_name,
            "max_bank": self.max_bank,
            "memory": self.memory.detach().cpu(),
            "fmap_hw": self.fmap_hw,
            "threshold": self.threshold,
            "train_mean": self.train_mean,
            "train_std": self.train_std,
        }

    def save(self, path: str):
        torch.save(self.state(), path)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "AnomalyModel":
        if not _TORCH_OK:
            raise RuntimeError("torch/torchvision kurulu degil.")
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.load(path, map_location=dev, weights_only=False)
        m = cls(input_size=data["input_size"], backbone=data["backbone"],
                max_bank=data["max_bank"], device=dev)
        m.memory = data["memory"].to(dev)
        m.fmap_hw = tuple(data["fmap_hw"])
        m.threshold = float(data["threshold"])
        m.train_mean = float(data["train_mean"])
        m.train_std = float(data["train_std"])
        return m
