import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import logging
import matplotlib.pyplot as plt

from .eval import clean_evaluations


# =========================
# 🔍 Visualization
# =========================
def visualize_predictions(model, loader, device, num_samples=3):
    model.eval()
    shown = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.4).float()

            imgs = imgs.cpu()
            masks = masks.cpu()
            preds = preds.cpu()

            for i in range(imgs.shape[0]):
                if shown >= num_samples:
                    return

                img = imgs[i].permute(1, 2, 0)
                mask = masks[i][0]
                pred = preds[i][0]

                plt.figure(figsize=(10, 3))

                plt.subplot(1, 3, 1)
                plt.title("Image")
                plt.imshow(img)
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.title("GT Mask")
                plt.imshow(mask, cmap="gray")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.title("Prediction")
                plt.imshow(pred, cmap="gray")
                plt.axis("off")

                plt.show()
                shown += 1


# =========================
# 📉 Loss + Metric
# =========================
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def dice_score(preds, targets, eps=1e-6):
    # 🔥 CHANGE: Proper batch-wise Dice
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


# =========================
# ⚙️ Helpers
# =========================
def _resolve_loader(dataset):
    if isinstance(dataset, DataLoader):
        return dataset
    if hasattr(dataset, "get_loader"):
        return dataset.get_loader()
    if hasattr(dataset, "__iter__"):
        return dataset
    raise ValueError("Invalid dataset")


def _resolve_dataset(dataset):
    return dataset.dataset if hasattr(dataset, "dataset") else dataset


def _loader_settings(dataset):
    return (
        getattr(dataset, "batch_size", 16),
        getattr(dataset, "shuffle", True),
        getattr(dataset, "num_workers", 4),
        getattr(dataset, "pin_memory", False),
    )


def _make_loader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):
    return DataLoader(dataset, batch_size, shuffle, num_workers, pin_memory)


# =========================
# 🧪 TEST EPOCH
# =========================
def _test_epoch(model, loader, criterion, device):
    # 🧼 CLEANUP: removed duplicate function
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            # 🔥 CHANGE: weighted loss
            bce = criterion(logits, masks)
            dice = dice_loss(logits, masks)
            loss = 0.75 * bce + 0.25 * dice
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.55).float()
            total_dice += dice_score(preds, masks).item() 

    return total_loss / len(loader), total_dice / len(loader)


# =========================
# 🏋️ TRAIN EPOCH
# =========================
def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        # 🔥 CHANGE: weighted loss
        bce = criterion(logits, masks)
        dice = dice_loss(logits, masks)
        loss = 0.75 * bce + 0.25 * dice

        optimizer.zero_grad()
        loss.backward()

        # 🔥 CHANGE: Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.55).float()
            total_dice += dice_score(preds, masks).item()

    return total_loss / len(loader), total_dice / len(loader)


# =========================
# 🚀 MAIN TRAIN
# =========================
def train_model_clean(model, train, test=None, device="cpu", epochs=5, lr=1e-3):

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler("training.log"),  # 🔥 CHANGE
            logging.StreamHandler()
        ],
        force=True
    )

    train_loader = _resolve_loader(train)
    test_loader = _resolve_loader(test) if test else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5, verbose=True
    )
    
    pos_weight = torch.tensor([2.5]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    logging.info("Starting training...\n")  # 🔥 nice header

    for epoch in range(epochs):
        train_loss, train_dice = _train_epoch(model, train_loader, optimizer, criterion, device)

        if test_loader:
            test_loss, test_dice = _test_epoch(model, test_loader, criterion, device)

            scheduler.step(test_loss)

            logging.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Train Dice: {train_dice:.4f} | "
                f"Test Dice: {test_dice:.4f}"
            )

    if test_loader:
        final_dice = clean_evaluations(model, test_loader, device, save_dir="train_eval")
        logging.info(f"\n[FINAL] Dice: {final_dice:.4f}")
        print(f"\n[FINAL] Dice: {final_dice:.4f}")

# =========================
# ⚔️ ADV TRAIN (unchanged)
# =========================
def train_model_adv(model, dataset, adv_buffer, device="cpu", epochs=3, lr=1e-3):
    clean_dataset = _resolve_dataset(dataset)

    if adv_buffer:
        adv_dataset = TensorDataset(*zip(*adv_buffer))
        combined_dataset = ConcatDataset([clean_dataset, adv_dataset])
    else:
        combined_dataset = clean_dataset

    batch_size, shuffle, num_workers, pin_memory = _loader_settings(dataset)

    loader = _make_loader(combined_dataset, batch_size, shuffle, num_workers, pin_memory)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        loss, dice = _train_epoch(model, loader, optimizer, criterion, device)
        print(f"ADV Epoch {epoch+1}/{epochs} | Dice: {dice:.4f}")