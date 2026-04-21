# safebooru — Danbooru 태그 분류 모델

Danbooru parquet 데이터셋을 이용한 멀티레이블 이미지 태깅 모델 학습 프로젝트입니다.

## 프로젝트 구조

```text
safebooru/
├── main.py                      # 진입점
├── requirements_train.txt       # 학습용 패키지 목록
├── .env                         # 계정 정보 (git 제외)
├── src/
│   ├── config.py                # 경로 및 하이퍼파라미터
│   ├── preprocess.py            # 태그맵 생성, val set 분리
│   ├── dataset.py               # Dataset, DataLoader, 이미지 다운로드
│   ├── model.py                 # 모델 빌드, 체크포인트 저장/복구
│   ├── loss.py                  # AsymmetricLoss
│   ├── train.py                 # 학습 루프
│   └── evaluate.py              # 검증 루프
└── data/
    ├── parquet/danbooru.parquet # 메인 데이터셋
    ├── model/                   # 태그맵, 체크포인트, 최적 모델 저장
    ├── cache/                   # val set 캐시
    └── images/
        ├── train/               # 학습 이미지 (임시)
        └── val/                 # 검증 이미지
```

## 설치

```bash
pip install -r requirements_train.txt
```
## 실행

```bash
python main.py
```

최초 실행 시 순서대로 진행됩니다:

1. parquet 전체 스캔 → 빈도 기준 태그맵 생성 (`data/model/tag_to_idx.json`)
2. 고정 val set 샘플링 및 저장 (`data/cache/val_set.parquet`)
3. 청크 단위로 이미지 다운로드 → 학습 → 검증 반복
4. 체크포인트 자동 저장 (`data/model/checkpoint.pth`)
5. val mAP 갱신 시 최적 모델 저장 (`data/model/best_model.pth`)

재실행 시 체크포인트가 있으면 이어서 학습합니다.

## CPU/GPU 자동 전환

| 항목 | GPU | CPU |
| --- | --- | --- |
| 모델 | convnextv2_base | convnextv2_tiny |
| 이미지 크기 | 224×224 | 224×224 |
| AMP (autocast) | 사용 | 미사용 |

## 주요 하이퍼파라미터

`src/config.py`에서 조정 가능합니다.

| 파라미터 | 기본값 | 설명 |
| --- | --- | --- |
| `TAG_MIN_FREQ` | 1000 | 태그 최소 등장 빈도 |
| `TRAIN_CHUNK_SIZE` | 5000 | 청크당 학습 이미지 수 |
| `VAL_SET_SIZE` | 1000 | 검증 이미지 수 |
| `REAL_BATCH_SIZE` | 4 | 배치 크기 |
| `ACCUMULATION_STEPS` | 8 | 그래디언트 누적 (실질 배치 = 32) |
| `LR` | 1e-4 | 학습률 |
| `PATIENCE` | 10 | Early stopping 기준 |
