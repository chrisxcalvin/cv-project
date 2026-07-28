# OWNER: toy training script (proof-of-concept only)
"""
Trains a toy MobileNetV2 real/spoof classifier so core/passive_classifier.py's
CNN branch has a real checkpoint to load instead of always falling back to
the rule-based path.

IMPORTANT - what this is and isn't:
  - "Real" class = LFW - Labeled Faces in the Wild (public research dataset,
    real COLOR photographs with varied lighting/pose/background, downloaded
    via sklearn - no signup/license gate). Earlier v1 of this script used
    Olivetti Faces (grayscale) and collapsed every image to R=G=B before
    training, so the model had never once seen real chromatic skin tones -
    it scored actual webcam photos near 50% (pure guessing) because they
    were completely out of its training distribution. LFW in full color,
    plus brightness/contrast jitter, closes most of that gap.
  - "Spoof" class = the SAME photos, synthetically degraded to mimic what a
    print or screen replay does to an image: JPEG recompression artifacts,
    a halftone dot overlay, and a downsample/upsample resolution loss. These
    are NOT real presentation-attack photos (no actual printed photo or
    screen was captured), so this model has still not been validated against
    real spoof attempts - it only proves the CNN code path (model file ->
    load -> inference) is wired correctly end-to-end, now on inputs that at
    least resemble what a webcam actually captures.

Run from the project root:
    .venv\\Scripts\\python.exe scripts/train_toy_model.py
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMG_SIZE = 224
MAX_BASE_PHOTOS = 300
JITTER_VARIANTS_PER_PHOTO = 1  # +1 brightness/contrast-jittered copy per photo
MODEL_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "passive_spoof_model.pt")


def jitter_lighting(rgb_uint8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random brightness/contrast jitter - webcam lighting varies far more
    than LFW's studio-ish photos, so this adds cheap lighting diversity."""
    alpha = rng.uniform(0.7, 1.3)  # contrast
    beta = rng.uniform(-30, 30)    # brightness
    return cv2.convertScaleAbs(rgb_uint8, alpha=alpha, beta=beta)


def make_spoof_version(rgb_uint8: np.ndarray) -> np.ndarray:
    """Degrade a real color face image to mimic print/replay artifacts."""
    h, w = rgb_uint8.shape[:2]

    # Downsample/upsample -> blockiness, like a screen replay resolution loss.
    small = cv2.resize(rgb_uint8, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR)
    blocky = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Halftone dot overlay -> mimics printed-photo dot pattern (broadcast
    # the same periodic mask across all 3 color channels).
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dot_period = 4
    halftone_mask = (((xx % dot_period) < dot_period // 2) ^
                      ((yy % dot_period) < dot_period // 2)).astype(np.float32)
    halftone_mask = halftone_mask[:, :, None]
    halftoned = blocky.astype(np.float32) * (0.75 + 0.25 * halftone_mask)
    halftoned = np.clip(halftoned, 0, 255).astype(np.uint8)

    # Heavy JPEG recompression -> compression-artifact high-frequency noise.
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(halftoned, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 15])
    if ok:
        spoof_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        spoof = cv2.cvtColor(spoof_bgr, cv2.COLOR_BGR2RGB)
    else:
        spoof = halftoned
    return spoof


def build_dataset():
    print("[train_toy_model] Downloading LFW color faces (~200MB, one-time)...")
    data = fetch_lfw_people(color=True, resize=1.0, min_faces_per_person=20)
    images = data.images  # (n, h, w, 3), float, range [0, 255]
    images = np.clip(images, 0, 255).astype(np.uint8)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(images), size=min(MAX_BASE_PHOTOS, len(images)), replace=False)
    base_photos = images[idx]

    real_imgs, spoof_imgs = [], []
    for photo in base_photos:
        resized = cv2.resize(photo, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
        variants = [resized]
        for _ in range(JITTER_VARIANTS_PER_PHOTO):
            variants.append(jitter_lighting(resized, rng))

        for variant in variants:
            real_imgs.append(variant)
            spoof_imgs.append(make_spoof_version(variant))

    X = real_imgs + spoof_imgs
    # Label convention must match core/passive_classifier.py::_cnn_predict,
    # which reads probs[0] as "real": class 0 = real, class 1 = spoof.
    y = [0] * len(real_imgs) + [1] * len(spoof_imgs)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


class FaceSpoofDataset(Dataset):
    def __init__(self, rgb_images, labels):
        self.images = rgb_images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx]), self.labels[idx]


def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


def main():
    X_train, X_val, y_train, y_val = build_dataset()
    print(f"[train_toy_model] train={len(X_train)} val={len(X_val)}")

    train_ds = FaceSpoofDataset(X_train, y_train)
    val_ds = FaceSpoofDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = build_model()

    # Freeze backbone, fine-tune the classifier head only - all that's
    # affordable on CPU in a few minutes for this toy dataset size.
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 6
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                out = model(imgs)
                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"[train_toy_model] epoch {epoch + 1}/{epochs} "
              f"train_loss={total_loss / len(train_ds):.4f} val_acc={correct / total:.3f}")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, MODEL_OUT)
    print(f"[train_toy_model] Saved checkpoint to {MODEL_OUT}")


if __name__ == "__main__":
    main()
