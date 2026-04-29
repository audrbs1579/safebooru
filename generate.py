"""
Henreader Style LoRA - 이미지 생성 스크립트
Base: Stable Diffusion anything-v5 + LoRA (pytorch_lora_weights.safetensors)
WD14 태거로 input/ 이미지에서 태그 자동 추출 후 img2img 변환

사용법:
  python generate.py                          # input/ 이미지 자동 처리
  python generate.py --prompt "1girl, solo"   # 커스텀 프롬프트 (태거 무시)
  python generate.py --count 4               # 여러 장 생성
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from PIL import Image


LORA_PATH = Path(__file__).parent / "data/model/pytorch_lora_weights.safetensors"
BASE_MODEL = "stablediffusionapi/anything-v5"
INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

TRIGGER = "henreader style"
DEFAULT_PROMPT = "henreader style, 1girl, solo, looking at viewer, high quality, masterpiece"
NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality"
)


# ── WD14 태거 ──────────────────────────────────────────────────────────────────

def load_wd14_tagger():
    print("WD14 태거 모델 다운로드 중...")
    repo_id = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
    model_path = hf_hub_download(repo_id, "model.onnx")
    tags_path = hf_hub_download(repo_id, "selected_tags.csv")

    tags = pd.read_csv(tags_path)["name"].tolist()
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session, tags


def tag_image(session, tags, image: Image.Image, threshold: float = 0.35) -> str:
    target = 448
    img = image.convert("RGB")
    img.thumbnail((target, target), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target, target), (255, 255, 255))
    canvas.paste(img, ((target - img.width) // 2, (target - img.height) // 2))

    arr = np.array(canvas, dtype=np.float32)[:, :, ::-1]
    arr = np.expand_dims(arr, axis=0)

    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: arr})[0][0]

    results = [
        (tags[i], preds[i]) for i in range(4, len(preds)) if preds[i] > threshold
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return ", ".join(tag for tag, _ in results)


# ── SD 파이프라인 ──────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
        return "cuda", torch.float16
    print("GPU 없음 - CPU 사용 (속도가 느릴 수 있습니다)")
    return "cpu", torch.float32


def build_txt2img_pipeline(device: str, dtype: torch.dtype):
    print("txt2img 파이프라인 로딩 중...")
    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype, safety_checker=None)
    pipe.load_lora_weights(str(LORA_PATH))
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe


def build_img2img_pipeline(device: str, dtype: torch.dtype):
    print("img2img 파이프라인 로딩 중...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype, safety_checker=None)
    pipe.load_lora_weights(str(LORA_PATH))
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe


def generate_txt2img(pipe, prompt: str, args) -> list[Image.Image]:
    full_prompt = prompt if TRIGGER in prompt else f"{TRIGGER}, {prompt}"
    print(f"프롬프트: {full_prompt}")
    return pipe(
        prompt=full_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=512,
        height=512,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        num_images_per_prompt=args.count,
        cross_attention_kwargs={"scale": args.lora_weight},
    ).images


def generate_img2img(pipe, input_image: Image.Image, prompt: str, args) -> list[Image.Image]:
    full_prompt = prompt if TRIGGER in prompt else f"{TRIGGER}, {prompt}"
    print(f"프롬프트: {full_prompt}")
    input_image = input_image.convert("RGB").resize((512, 512), Image.LANCZOS)
    return pipe(
        prompt=full_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=input_image,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        num_images_per_prompt=args.count,
        cross_attention_kwargs={"scale": args.lora_weight},
    ).images


def save_images(images: list[Image.Image], prefix: str = "output"):
    OUTPUT_DIR.mkdir(exist_ok=True)
    for img in images:
        idx = 0
        while True:
            name = f"{prefix}.png" if idx == 0 else f"{prefix}_{idx:03d}.png"
            path = OUTPUT_DIR / name
            if not path.exists():
                break
            idx += 1
        img.save(path)
        print(f"저장: {path}")


def collect_input_images() -> list[Path]:
    INPUT_DIR.mkdir(exist_ok=True)
    return sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Henreader Style LoRA 이미지 생성")
    parser.add_argument("--prompt", type=str, default=None, help="고정 프롬프트 (미지정시 WD14 태거 자동 추출)")
    parser.add_argument("--count", type=int, default=1, help="생성할 이미지 수")
    parser.add_argument("--steps", type=int, default=25, help="샘플링 스텝 수 (기본 25)")
    parser.add_argument("--cfg", type=float, default=7.0, help="CFG Scale (기본 7.0)")
    parser.add_argument("--lora_weight", type=float, default=0.8, help="LoRA 가중치 0.6~0.9 (기본 0.8)")
    parser.add_argument("--strength", type=float, default=0.75, help="img2img 변형 강도 0~1 (기본 0.75)")
    parser.add_argument("--threshold", type=float, default=0.35, help="WD14 태그 임계값 (기본 0.35)")
    args = parser.parse_args()

    if not LORA_PATH.exists():
        print(f"오류: LoRA 파일을 찾을 수 없습니다 - {LORA_PATH}")
        return

    device, dtype = get_device()
    input_images = collect_input_images()

    if input_images:
        print(f"input/ 에서 {len(input_images)}개 이미지 발견 → img2img 모드")
        tagger_session, tag_list = load_wd14_tagger()
        pipe = build_img2img_pipeline(device, dtype)

        for input_path in input_images:
            print(f"\n처리 중: {input_path.name}")
            image = Image.open(input_path)

            if args.prompt:
                prompt = args.prompt
            else:
                extracted = tag_image(tagger_session, tag_list, image, args.threshold)
                prompt = extracted
                print(f"추출된 태그: {extracted}")

            images = generate_img2img(pipe, image, prompt, args)
            save_images(images, prefix=input_path.stem)
    else:
        print("input/ 이미지 없음 → txt2img 모드")
        pipe = build_txt2img_pipeline(device, dtype)
        prompt = args.prompt if args.prompt else DEFAULT_PROMPT
        images = generate_txt2img(pipe, prompt, args)
        save_images(images, prefix="henreader")

    print("\n완료!")


if __name__ == "__main__":
    main()
