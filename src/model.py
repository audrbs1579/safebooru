"""모델 생성 및 체크포인트 저장/복구."""
import os
import torch
import timm

from config import DEVICE, LR, WEIGHT_DECAY, MAX_STEPS, CHECKPOINT_PATH, BEST_MODEL_PATH


def build_model(num_classes: int):
    # CPU 환경에서는 가벼운 모델 사용
    backbone = 'convnextv2_tiny' if not torch.cuda.is_available() else 'convnextv2_base'
    model = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
    return model.to(DEVICE)


def build_optimizer(model: torch.nn.Module):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def build_scheduler(optimizer, last_epoch: int = -1):
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    return CosineAnnealingWarmRestarts(optimizer, T_0=MAX_STEPS, T_mult=1, eta_min=1e-6)


def save_checkpoint(path: str, model, optimizer, scheduler, state: dict):
    payload = {
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    payload.update(state)
    torch.save(payload, path)


def load_checkpoint(model, optimizer, scheduler):
    """체크포인트가 있으면 로드하고 저장된 상태 dict를 반환한다."""
    if not os.path.exists(CHECKPOINT_PATH):
        print("처음부터 학습 시작")
        return {}

    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    state = {k: v for k, v in ckpt.items()
             if k not in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict')}
    print(
        f"체크포인트 복구 | rows: {state.get('processed_rows', 0):,} "
        f"| step: {state.get('global_step', 0):,} "
        f"| best mAP: {state.get('best_map', -1):.4f}"
    )
    return state
