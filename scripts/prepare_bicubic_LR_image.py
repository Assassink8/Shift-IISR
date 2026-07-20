import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

import utils.util_sisr as util_sisr
import utils.util_image as util_image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def pil_to_float01_hwc(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0


def float01_hwc_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr_u8, mode="RGB")


def process_one(in_path: Path, lr_path: Path, scale: float, min_max=(0.0, 1.0)):
    img = Image.open(in_path)
    x = pil_to_float01_hwc(img)

    bic = util_sisr.Bicubic(scale=scale, out_shape=None)
    clamp = util_image.Clamper(min_max=min_max)

    x_lr = bic(x)
    x_lr = clamp(x_lr)

    lr_path.parent.mkdir(parents=True, exist_ok=True)
    float01_hwc_to_pil(x_lr).save(lr_path)

    return x.shape[:2], x_lr.shape[:2]


def main():
    parser = argparse.ArgumentParser(description="Only generate LR images with bicubic downsampling.")
    parser.add_argument("--in_dir", type=str, required=True, help="输入原图文件夹")
    parser.add_argument("--lr_dir", type=str, required=True, help="LR输出文件夹")
    parser.add_argument("--scale", type=float, default=0.25, help="下采样比例，例如 4倍下采样填 0.25")
    parser.add_argument("--lr_suffix", type=str, default="", help="LR 文件名后缀")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    lr_dir = Path(args.lr_dir)

    if not in_dir.exists():
        raise FileNotFoundError(f"in_dir not found: {in_dir}")

    files = [p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    print(f"Found {len(files)} images in {in_dir}")

    ok_count = 0
    fail_count = 0

    for p in tqdm(files, desc="Processing", unit="img"):
        rel = p.relative_to(in_dir)
        lr_name = rel.stem + args.lr_suffix + rel.suffix
        lr_path = lr_dir / rel.with_name(lr_name)

        try:
            orig_hw, lr_hw = process_one(
                in_path=p,
                lr_path=lr_path,
                scale=args.scale,
                min_max=(0.0, 1.0),
            )
            ok_count += 1
            # print(f"[OK] {p.name}: orig={orig_hw[1]}x{orig_hw[0]} -> lr={lr_hw[1]}x{lr_hw[0]}")
        except Exception as e:
            fail_count += 1
            print(f"[WARN] failed: {p} -> {e}")

    print("-" * 60)
    print(f"LR dir: {lr_dir}")
    print(f"Success: {ok_count}")
    print(f"Failed : {fail_count}")


if __name__ == "__main__":
    main()
