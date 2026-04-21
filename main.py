"""진입점. python main.py 로 실행."""
import sys
import os

# .env 파일 로드
_env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(_env_path):
    with open(_env_path, encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# src 패키지를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import get_tag_map, get_val_set
from model import build_model, build_optimizer, build_scheduler, load_checkpoint
from loss import AsymmetricLoss
from train import train
from config import DEVICE

def main():
    print(f"Device: {DEVICE}")

    tag_to_idx = get_tag_map()
    num_classes = len(tag_to_idx)
    print(f"클래스 수: {num_classes:,}")

    val_df = get_val_set()
    print(f"Val set: {len(val_df):,}장")

    model     = build_model(num_classes)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    criterion = AsymmetricLoss()

    state = load_checkpoint(model, optimizer, scheduler)

    train(model, optimizer, scheduler, criterion, val_df, tag_to_idx, state)


if __name__ == '__main__':
    main()
