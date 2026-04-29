"""
Henreader Style LoRA - 이미지 생성 스크립트
Base: Stable Diffusion v1.5 + LoRA (pytorch_lora_weights.safetensors)

사용법:
  python generate.py                          # 기본 프롬프트로 생성
  python generate.py --prompt "1girl, solo"   # 커스텀 프롬프트
  python generate.py --input image.png        # 입력 이미지 기반 img2img
  python generate.py --count 4               # 여러 장 생성
"""

import argparse
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image


LORA_PATH = Path(__file__).parent / "data/model/pytorch_lora_weights.safetensors"
BASE_MODEL = "stablediffusionapi/anything-v5"
INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"

TRIGGER = "henreader style"
DEFAULT_PROMPT = "henreader style, 1girl, solo, looking at viewer, high quality, masterpiece"
NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality"
)


def get_device():
    if torch.cuda.is_available():
        print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
        return "cuda", torch.float16
    print("GPU 없음 - CPU 사용 (속도가 느릴 수 있습니다)")
    return "cpu", torch.float32


def build_txt2img_pipeline(device: str, dtype: torch.dtype):
    print("파이프라인 로딩 중...")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.load_lora_weights(str(LORA_PATH))
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe


def build_img2img_pipeline(device: str, dtype: torch.dtype):
    print("img2img 파이프라인 로딩 중...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.load_lora_weights(str(LORA_PATH))
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe


def generate_txt2img(pipe, prompt: str, args) -> list[Image.Image]:
    full_prompt = prompt if TRIGGER in prompt else f"{TRIGGER}, {prompt}"
    print(f"프롬프트: {full_prompt}")

    images = pipe(
        prompt=full_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=512,
        height=512,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        num_images_per_prompt=args.count,
        cross_attention_kwargs={"scale": args.lora_weight},
    ).images
    return images


def generate_img2img(pipe, input_image: Image.Image, prompt: str, args) -> list[Image.Image]:
    full_prompt = prompt if TRIGGER in prompt else f"{TRIGGER}, {prompt}"
    print(f"프롬프트: {full_prompt}")

    # 512x512 리사이즈 (비율 유지 후 크롭)
    input_image = input_image.convert("RGB").resize((512, 512), Image.LANCZOS)

    images = pipe(
        prompt=full_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=input_image,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        num_images_per_prompt=args.count,
        cross_attention_kwargs={"scale": args.lora_weight},
    ).images
    return images


def save_images(images: list[Image.Image], prefix: str = "output"):
    OUTPUT_DIR.mkdir(exist_ok=True)
    saved = []
    for i, img in enumerate(images):
        # 이미 같은 이름이 있으면 번호 증가
        idx = 0
        while True:
            name = f"{prefix}_{idx:03d}.png" if len(images) > 1 or idx > 0 else f"{prefix}.png"
            path = OUTPUT_DIR / name
            if not path.exists():
                break
            idx += 1
        img.save(path)
        print(f"저장: {path}")
        saved.append(path)
    return saved


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def collect_input_images() -> list[Path]:
    if not INPUT_DIR.exists():
        INPUT_DIR.mkdir(exist_ok=True)
    images = sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return images


def main():
    parser = argparse.ArgumentParser(description="Henreader Style LoRA 이미지 생성")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="생성 프롬프트")
    parser.add_argument("--count", type=int, default=1, help="생성할 이미지 수")
    parser.add_argument("--steps", type=int, default=25, help="샘플링 스텝 수 (기본 25)")
    parser.add_argument("--cfg", type=float, default=7.0, help="CFG Scale (기본 7.0)")
    parser.add_argument("--lora_weight", type=float, default=0.8, help="LoRA 가중치 0.6~0.9 (기본 0.8)")
    parser.add_argument("--strength", type=float, default=0.75, help="img2img 변형 강도 0~1 (기본 0.75)")
    args = parser.parse_args()

    if not LORA_PATH.exists():
        print(f"오류: LoRA 파일을 찾을 수 없습니다 - {LORA_PATH}")
        return

    device, dtype = get_device()
    input_images = collect_input_images()

    if input_images:
        print(f"input/ 에서 {len(input_images)}개 이미지 발견 → img2img 모드")
        pipe = build_img2img_pipeline(device, dtype)
        for input_path in input_images:
            print(f"\n처리 중: {input_path.name}")
            input_image = Image.open(input_path)
            images = generate_img2img(pipe, input_image, args.prompt, args)
            save_images(images, prefix=input_path.stem)
    else:
        print("input/ 이미지 없음 → txt2img 모드")
        pipe = build_txt2img_pipeline(device, dtype)
        images = generate_txt2img(pipe, args.prompt, args)
        save_images(images, prefix="henreader")

    print("\n완료!")


if __name__ == "__main__":
    main()
