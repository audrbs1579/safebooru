"""태그맵 생성 및 고정 Validation set 분리."""
import json
import gc
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from config import (
    PARQUET_PATH, TAGMAP_PATH, VAL_CACHE_PATH,
    TAG_MIN_FREQ, MIN_IMG_SIZE, READ_BATCH_SIZE, VAL_SET_SIZE,
)


def _base_filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df['is_deleted'] == False) &
        (df['is_banned']  == False) &
        (df['image_width']  >= MIN_IMG_SIZE) &
        (df['image_height'] >= MIN_IMG_SIZE)
    ]


def build_tag_map() -> dict:
    """parquet 전체를 스캔해 빈도 기준 태그맵을 생성하고 저장한다."""
    print("=== 태그 스캔 시작 ===")
    counter: Counter = Counter()
    pf = pq.ParquetFile(PARQUET_PATH)

    for batch in tqdm(pf.iter_batches(batch_size=READ_BATCH_SIZE), desc="태그 분석"):
        df = _base_filter(batch.to_pandas())
        for tags_str in df['tag_string'].dropna():
            counter.update(str(tags_str).split())
        del df
        gc.collect()

    filtered = sorted(t for t, c in counter.items() if c >= TAG_MIN_FREQ)
    tag_to_idx = {tag: idx for idx, tag in enumerate(filtered)}

    with open(TAGMAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(tag_to_idx, f, ensure_ascii=False)

    print(f"태그맵 저장 완료 → {TAGMAP_PATH}  (클래스 수: {len(tag_to_idx):,})")
    return tag_to_idx


def load_tag_map() -> dict:
    with open(TAGMAP_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_val_set() -> pd.DataFrame:
    """전체 parquet에서 균등 샘플링으로 고정 val set을 생성하고 저장한다."""
    print("=== Val set 샘플링 ===")
    pf = pq.ParquetFile(PARQUET_PATH)
    samples = []
    total_rows = 0

    for batch in tqdm(pf.iter_batches(batch_size=READ_BATCH_SIZE), desc="Val 샘플링"):
        df = _base_filter(batch.to_pandas())
        if len(df) == 0:
            continue
        total_rows += len(df)
        n_pick = max(1, int(len(df) * VAL_SET_SIZE / 7_000_000))
        samples.append(df.sample(n=min(n_pick, len(df)), random_state=42))
        if sum(len(s) for s in samples) >= VAL_SET_SIZE * 2:
            break

    val_df = (
        pd.concat(samples, ignore_index=True)
        .sample(n=min(VAL_SET_SIZE, sum(len(s) for s in samples)), random_state=42)
        .reset_index(drop=True)
    )
    val_df.to_parquet(VAL_CACHE_PATH)
    print(f"Val set 저장 → {VAL_CACHE_PATH}  ({len(val_df):,}장)")
    return val_df


def load_val_set() -> pd.DataFrame:
    return pd.read_parquet(VAL_CACHE_PATH)


def get_tag_map() -> dict:
    """태그맵이 없으면 생성, 있으면 로드."""
    import os
    return load_tag_map() if os.path.exists(TAGMAP_PATH) else build_tag_map()


def get_val_set() -> pd.DataFrame:
    """val set이 없으면 생성, 있으면 로드."""
    import os
    return load_val_set() if os.path.exists(VAL_CACHE_PATH) else build_val_set()
