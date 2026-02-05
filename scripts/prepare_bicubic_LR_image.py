import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# 你项目里的这俩模块（按你的工程路径改一下 import）
import utils.util_sisr as util_sisr
import utils.util_image as util_image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def pil_to_float01_hwc(pil_img: Image.Image) -> np.ndarray:
    """PIL RGB -> float32 HWC in [0,1]."""
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def float01_hwc_to_pil(arr: np.ndarray) -> Image.Image:
    """float32 HWC in [0,1] -> PIL RGB."""
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr_u8, mode="RGB")


def process_one(in_path: Path, out_path: Path, scale: float, min_max=(0.0, 1.0)):
    # 读图
    img = Image.open(in_path)
    x = pil_to_float01_hwc(img)  # HWC, [0,1]

    # bicubic_norm 的核心：Bicubic -> Clamper
    bic = util_sisr.Bicubic(scale=scale, out_shape=None)
    clamp = util_image.Clamper(min_max=min_max)

    y = bic(x)      # 期望输出还是 HWC, float
    y = clamp(y)    # clip 到 [0,1]

    # 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    float01_hwc_to_pil(y).save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True, help="输入文件夹")
    parser.add_argument("--out_dir", type=str, required=True, help="输出文件夹")
    parser.add_argument("--scale", type=float, default=0.25, help="缩放比例，1/4=0.25")
    parser.add_argument("--suffix", type=str, default="_lr", help="输出文件名后缀")
    parser.add_argument("--keep_ext", action="store_true", help="保持原扩展名（默认保持）")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    scale = args.scale

    if not in_dir.exists():
        raise FileNotFoundError(f"in_dir not found: {in_dir}")

    files = [p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"Found {len(files)} images in {in_dir}")

    for p in files:
        rel = p.relative_to(in_dir)
        stem = rel.stem + args.suffix
        ext = rel.suffix  # 默认保持原扩展名
        out_path = out_dir / rel.with_name(stem + ext)

        try:
            process_one(p, out_path, scale=scale, min_max=(0.0, 1.0))
        except Exception as e:
            print(f"[WARN] failed: {p} -> {e}")

    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
