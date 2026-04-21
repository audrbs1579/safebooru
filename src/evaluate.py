"""검증 루프."""
import torch
from tqdm.auto import tqdm
from torchmetrics.classification import MultilabelAveragePrecision

from config import DEVICE


def evaluate(model: torch.nn.Module, val_loader, num_classes: int) -> float:
    metric = MultilabelAveragePrecision(num_labels=num_classes, average="macro").to(DEVICE)
    model.eval()
    metric.reset()

    with torch.no_grad():
        for data in tqdm(val_loader, desc="Val", leave=False):
            if data is None:
                continue
            imgs, tgs = data[0].to(DEVICE), data[1].to(DEVICE)

            # CPU에서는 autocast 없이 실행
            if DEVICE.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    preds = torch.sigmoid(model(imgs))
            else:
                preds = torch.sigmoid(model(imgs))

            metric.update(preds, tgs.long())

    try:
        return metric.compute().item()
    except Exception:
        return 0.0
