"""메인 학습 루프."""
import gc
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm.auto import tqdm

from config import (
    DEVICE, ACCUMULATION_STEPS, MAX_GRAD_NORM,
    LR, WARMUP_STEPS, PATIENCE,
    CHECKPOINT_PATH, BEST_MODEL_PATH,
    IMG_DIR, VAL_IMG_DIR,
)
from dataset import (
    chunk_generator, download_chunk, make_loader,
    train_transform, eval_transform,
)
from evaluate import evaluate
from model import save_checkpoint


def _cleanup_dir(directory: str, keep_ids: set = None):
    for fpath in glob.glob(os.path.join(directory, '*')):
        pid = os.path.splitext(os.path.basename(fpath))[0]
        if keep_ids is None or pid not in keep_ids:
            try:
                os.remove(fpath)
            except Exception:
                pass


def train(model, optimizer, scheduler, criterion, val_df, tag_to_idx, state: dict):
    num_classes = len(tag_to_idx)
    best_map         = state.get('best_map', -1.0)
    patience_counter = state.get('patience_counter', 0)
    global_step      = state.get('global_step', 0)
    start_row_idx    = state.get('processed_rows', 0)

    # CPU는 GradScaler 불필요
    use_amp = DEVICE.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    val_ids = set(val_df['post_id'].tolist())

    # Val 이미지 다운로드 (최초 1회)
    already = len(glob.glob(os.path.join(VAL_IMG_DIR, '*')))
    if already < len(val_df) * 0.8:
        print("Val 이미지 다운로드 중...")
        stats = download_chunk(val_df.to_dict('records'), VAL_IMG_DIR, desc="Val DL")
        print(f"  → ok {stats['ok']:,} / fail {stats['fail']:,}")

    val_loader = make_loader(val_df, VAL_IMG_DIR, tag_to_idx, eval_transform, shuffle=False)

    chunk_gen = chunk_generator(exclude_ids=val_ids)

    # start_row_idx까지 청크 건너뛰기
    rows_seen = 0
    first_chunk = None
    for chunk in chunk_gen:
        if rows_seen + len(chunk) <= start_row_idx:
            rows_seen += len(chunk)
            continue
        first_chunk = chunk
        rows_seen += len(chunk)
        break

    if first_chunk is None:
        print("처리할 청크 없음. 종료.")
        return

    _cleanup_dir(IMG_DIR)
    print(f"첫 청크 다운로드 ({len(first_chunk):,}장)...")
    download_chunk(first_chunk.to_dict('records'), IMG_DIR, desc="DL(init)")
    current_chunk = first_chunk

    dl_executor = ThreadPoolExecutor(max_workers=1)

    try:
        next_chunk  = next(chunk_gen)
        next_future = dl_executor.submit(download_chunk, next_chunk.to_dict('records'), IMG_DIR, "DL(bg)")
    except StopIteration:
        next_chunk  = None
        next_future = None

    stop = False

    while current_chunk is not None and not stop:
        train_loader = make_loader(current_chunk, IMG_DIR, tag_to_idx, train_transform, shuffle=True)

        model.train()
        total_loss, n_batches = 0.0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Train rows={start_row_idx:,} step={global_step:,}")
        for i, data in enumerate(pbar):
            if data is None:
                continue
            imgs = data[0].to(DEVICE)
            tgs  = data[1].to(DEVICE)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(imgs), tgs) / ACCUMULATION_STEPS
                scaler.scale(loss).backward()
            else:
                loss = criterion(model(imgs), tgs) / ACCUMULATION_STEPS
                loss.backward()

            total_loss += loss.item() * ACCUMULATION_STEPS
            n_batches  += 1

            if (i + 1) % ACCUMULATION_STEPS == 0:
                # Warmup LR
                if global_step < WARMUP_STEPS:
                    lr_now = LR * (global_step + 1) / WARMUP_STEPS
                    for pg in optimizer.param_groups:
                        pg['lr'] = lr_now

                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

                optimizer.zero_grad()
                if global_step >= WARMUP_STEPS:
                    scheduler.step()
                global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item() * ACCUMULATION_STEPS:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        avg_loss = total_loss / max(n_batches, 1)
        current_map = evaluate(model, val_loader, num_classes)
        start_row_idx += len(current_chunk)

        if current_map > best_map:
            best_map = current_map
            patience_counter = 0
            save_checkpoint(BEST_MODEL_PATH, model, optimizer, scheduler, {
                'processed_rows': start_row_idx, 'global_step': global_step,
                'best_map': best_map, 'patience_counter': patience_counter,
            })
            flag = f"best mAP {best_map:.4f} 갱신"
        else:
            patience_counter += 1
            flag = f"개선X ({patience_counter}/{PATIENCE})"

        print(
            f"[CHUNK] rows={start_row_idx:,} step={global_step:,} "
            f"loss={avg_loss:.4f} val_mAP={current_map:.4f} | {flag}"
        )

        save_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler, {
            'processed_rows': start_row_idx, 'global_step': global_step,
            'best_map': best_map, 'patience_counter': patience_counter,
        })

        if patience_counter >= PATIENCE:
            print(f"Early stopping (best mAP={best_map:.4f})")
            stop = True
            break

        # 다음 청크 다운로드 완료 대기
        if next_future is not None:
            stats = next_future.result()
            print(f"  [bg DL] ok {stats['ok']:,} / fail {stats['fail']:,}")

        # 다음 청크에 필요 없는 파일 삭제
        cur_ids  = {str(p) for p in current_chunk['post_id']}
        next_ids = {str(p) for p in next_chunk['post_id']} if next_chunk is not None else set()
        _cleanup_dir(IMG_DIR, keep_ids=next_ids)

        current_chunk = next_chunk
        try:
            next_chunk  = next(chunk_gen)
            next_future = dl_executor.submit(download_chunk, next_chunk.to_dict('records'), IMG_DIR, "DL(bg)")
        except StopIteration:
            next_chunk  = None
            next_future = None

        gc.collect()

    dl_executor.shutdown(wait=False)
    print(f"\n=== 학습 종료 | best mAP: {best_map:.4f} ===")
