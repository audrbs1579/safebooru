"""Dataset, DataLoader, 이미지 다운로드."""
import os
import gc
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from config import (
    PARQUET_PATH, IMG_SIZE, READ_BATCH_SIZE, TRAIN_CHUNK_SIZE,
    REAL_BATCH_SIZE, DOWNLOAD_WORKERS, DOWNLOAD_TIMEOUT, DOWNLOAD_RETRIES,
    DANBOORU_USERNAME, DANBOORU_API_KEY, MIN_IMG_SIZE,
)

try:
    from curl_cffi import requests as cffi_requests
    _USE_CURL = True
except ImportError:
    import requests as cffi_requests
    _USE_CURL = False


# ── Transform ──────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    v2.RandAugment(num_ops=2, magnitude=7),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])

eval_transform = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])


# ── Dataset ────────────────────────────────────────────────────────────────
class DanbooruDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, tag_to_idx: dict, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tag_to_idx = tag_to_idx
        self.num_classes = len(tag_to_idx)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['post_id']}.{row['file_ext']}")
        if not os.path.exists(img_path):
            return None
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            return None
        if self.transform:
            image = self.transform(image)

        labels = np.zeros(self.num_classes, dtype=np.float32)
        for tag in str(row['tag_string']).split():
            if tag in self.tag_to_idx:
                labels[self.tag_to_idx[tag]] = 1.0
        return image, torch.tensor(labels)


def safe_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def make_loader(df: pd.DataFrame, img_dir: str, tag_to_idx: dict,
                transform, shuffle: bool) -> DataLoader:
    return DataLoader(
        DanbooruDataset(df, img_dir, tag_to_idx, transform),
        batch_size=REAL_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,          # CPU 환경에서는 0이 안정적
        collate_fn=safe_collate_fn,
        pin_memory=False,
    )


# ── 다운로드 ───────────────────────────────────────────────────────────────
def _download_one(row, img_dir: str, stats: dict):
    md5 = str(row['md5'])
    ext = row['file_ext']
    url = f"https://cdn.donmai.us/original/{md5[:2]}/{md5[2:4]}/{md5}.{ext}"
    save_path = os.path.join(img_dir, f"{row['post_id']}.{ext}")

    if os.path.exists(save_path):
        stats['ok'] += 1
        return

    headers = {
        'User-Agent': 'DanbooruDL/1.0',
        'Referer': 'https://danbooru.donmai.us/',
    }
    kwargs = dict(headers=headers, timeout=DOWNLOAD_TIMEOUT)
    if DANBOORU_USERNAME and DANBOORU_API_KEY:
        kwargs['auth'] = (DANBOORU_USERNAME, DANBOORU_API_KEY)
    if _USE_CURL:
        kwargs['impersonate'] = 'chrome'

    for attempt in range(DOWNLOAD_RETRIES + 1):
        try:
            r = cffi_requests.get(url, **kwargs)
            if r.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(r.content)
                stats['ok'] += 1
                return
            if r.status_code in (403, 404, 410):
                break
        except Exception:
            pass
        if attempt < DOWNLOAD_RETRIES:
            time.sleep(0.5)

    stats['fail'] += 1


def download_chunk(records: list, img_dir: str, desc: str = "Download") -> dict:
    stats = {'ok': 0, 'fail': 0}
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        list(tqdm(
            ex.map(lambda r: _download_one(r, img_dir, stats), records),
            total=len(records), desc=desc, leave=False,
        ))
    return stats


# ── Chunk generator ────────────────────────────────────────────────────────
def chunk_generator(exclude_ids: set):
    """parquet을 청크 단위로 읽어 셔플 후 yield한다."""
    pf = pq.ParquetFile(PARQUET_PATH)
    buffer = []
    buf_size = 0

    for batch in pf.iter_batches(batch_size=READ_BATCH_SIZE):
        df = batch.to_pandas()
        df = df[
            (df['is_deleted'] == False) &
            (df['is_banned']  == False) &
            (df['image_width']  >= MIN_IMG_SIZE) &
            (df['image_height'] >= MIN_IMG_SIZE)
        ]
        df = df[~df['post_id'].isin(exclude_ids)]
        df = df.sample(frac=1).reset_index(drop=True)

        buffer.append(df)
        buf_size += len(df)

        while buf_size >= TRAIN_CHUNK_SIZE:
            big = pd.concat(buffer, ignore_index=True)
            chunk, rest = big.iloc[:TRAIN_CHUNK_SIZE], big.iloc[TRAIN_CHUNK_SIZE:]
            buffer = [rest]
            buf_size = len(rest)
            yield chunk
            gc.collect()

    if buf_size > 0:
        yield pd.concat(buffer, ignore_index=True)
