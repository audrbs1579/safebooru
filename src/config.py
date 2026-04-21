import os
import torch

# 경로
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH    = os.path.join(BASE_DIR, 'data', 'parquet', 'danbooru.parquet')
MODEL_DIR       = os.path.join(BASE_DIR, 'data', 'model')
CACHE_DIR       = os.path.join(BASE_DIR, 'data', 'cache')
IMG_DIR         = os.path.join(BASE_DIR, 'data', 'images', 'train')
VAL_IMG_DIR     = os.path.join(BASE_DIR, 'data', 'images', 'val')

CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'checkpoint.pth')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
TAGMAP_PATH     = os.path.join(MODEL_DIR, 'tag_to_idx.json')
VAL_CACHE_PATH  = os.path.join(CACHE_DIR, 'val_set.parquet')

for d in [MODEL_DIR, CACHE_DIR, IMG_DIR, VAL_IMG_DIR]:
    os.makedirs(d, exist_ok=True)

# 디바이스 — CPU 환경이면 자동으로 cpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 태그 필터
TAG_MIN_FREQ    = 1000
MIN_IMG_SIZE    = 256

# 데이터 로딩
READ_BATCH_SIZE  = 50000   # parquet 읽기 단위 (메모리 절약)
TRAIN_CHUNK_SIZE = 5000    # 한 번에 학습할 이미지 수 (CPU용 축소)
VAL_SET_SIZE     = 1000    # val set 크기 (CPU용 축소)

# 학습 하이퍼파라미터
REAL_BATCH_SIZE    = 4     # CPU면 8 → 4
ACCUMULATION_STEPS = 8     # effective batch = 32
IMG_SIZE           = 224   # CPU용 축소 (448 → 224)
LR                 = 1e-4
WEIGHT_DECAY       = 1e-2
MAX_GRAD_NORM      = 1.0
WARMUP_STEPS       = 200
MAX_STEPS          = 20000
PATIENCE           = 10

# 다운로드 설정
DOWNLOAD_WORKERS  = 10
DOWNLOAD_TIMEOUT  = 10
DOWNLOAD_RETRIES  = 2

# Danbooru 계정 (환경변수 또는 직접 입력)
DANBOORU_USERNAME = os.environ.get('DANBOORU_USERNAME', '')
DANBOORU_API_KEY  = os.environ.get('DANBOORU_API_KEY', '')
