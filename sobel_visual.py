#!/usr/bin/env python3
"""Compute Sobel x/y components and save visualization outputs.

Usage:
  python sobel_visual.py --input path/to/image.png
  python sobel_visual.py --input path/to/image.png --outdir output/sobel --ksize 3
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def normalize_abs_to_u8(arr: np.ndarray) -> np.ndarray:
    """Convert signed/float gradient map to uint8 for visualization."""
    abs_arr = np.abs(arr)
    max_val = float(abs_arr.max())
    if max_val < 1e-12:
        return np.zeros_like(abs_arr, dtype=np.uint8)
    return np.clip(abs_arr * (255.0 / max_val), 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Sobel x/y components and save visualization images."
    )
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument(
        "--outdir",
        default="output/sobel_visual",
        help="Directory to save outputs (default: output/sobel_visual)",
    )
    parser.add_argument(
        "--ksize",
        type=int,
        default=3,
        choices=[1, 3, 5, 7],
        help="Sobel kernel size (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gray = cv2.imread(str(in_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Failed to read image: {in_path}")

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=args.ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=args.ksize)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)

    # 16-bit signed preserves positive/negative gradient values.
    sobel_x_int16 = np.clip(np.round(sobel_x), -32768, 32767).astype(np.int16)
    sobel_y_int16 = np.clip(np.round(sobel_y), -32768, 32767).astype(np.int16)
    sobel_mag_u16 = np.clip(np.round(sobel_mag), 0, 65535).astype(np.uint16)

    vis_x = normalize_abs_to_u8(sobel_x)
    vis_y = normalize_abs_to_u8(sobel_y)
    vis_mag = normalize_abs_to_u8(sobel_mag)

    # Build side-by-side visualization: input | sobel_x | sobel_y | sobel_full
    vis_input = gray
    panel = np.hstack([vis_input, vis_x, vis_y, vis_mag])

    stem = in_path.stem
    x_path = outdir / f"{stem}_sobel_x.png"
    y_path = outdir / f"{stem}_sobel_y.png"
    full_path = outdir / f"{stem}_sobel_full.png"
    x_raw_path = outdir / f"{stem}_sobel_x_int16.png"
    y_raw_path = outdir / f"{stem}_sobel_y_int16.png"
    full_raw_path = outdir / f"{stem}_sobel_full_u16.png"
    panel_path = outdir / f"{stem}_sobel_panel.png"

    ok = True
    ok &= cv2.imwrite(str(x_path), vis_x)
    ok &= cv2.imwrite(str(y_path), vis_y)
    ok &= cv2.imwrite(str(full_path), vis_mag)
    ok &= cv2.imwrite(str(x_raw_path), sobel_x_int16)
    ok &= cv2.imwrite(str(y_raw_path), sobel_y_int16)
    ok &= cv2.imwrite(str(full_raw_path), sobel_mag_u16)
    ok &= cv2.imwrite(str(panel_path), panel)

    if not ok:
        raise RuntimeError("Failed to write one or more output images")

    print(f"Input: {in_path}")
    print(f"Saved: {x_path}")
    print(f"Saved: {y_path}")
    print(f"Saved: {full_path}")
    print(f"Saved (signed int16): {x_raw_path}")
    print(f"Saved (signed int16): {y_raw_path}")
    print(f"Saved (uint16): {full_raw_path}")
    print(f"Saved visualization panel: {panel_path}")


if __name__ == "__main__":
    main()
