#!/usr/bin/env python
import argparse
from pathlib import Path

import pyiqa
import torch
from tqdm import tqdm

from utils import util_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
METRIC_NAMES = ("psnr", "ssim", "lpips")


def list_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_image_pairs(result_dir, reference_dir, result_suffix="", ntest=None):
    result_images = list_images(result_dir)
    reference_images = list_images(reference_dir)
    reference_by_stem = {path.stem: path for path in reference_images}

    if len(reference_by_stem) != len(reference_images):
        raise ValueError("Reference filenames must have unique stems.")

    pairs = []
    missing_references = []
    for result_path in result_images:
        result_stem = result_path.stem
        if result_suffix:
            if not result_stem.endswith(result_suffix):
                missing_references.append(result_path.name)
                continue
            reference_stem = result_stem[:-len(result_suffix)]
        else:
            reference_stem = result_stem

        reference_path = reference_by_stem.get(reference_stem)
        if reference_path is None:
            missing_references.append(result_path.name)
            continue
        pairs.append((result_path, reference_path))

    if missing_references:
        preview = ", ".join(missing_references[:5])
        if len(missing_references) > 5:
            preview += ", ..."
        raise ValueError(
            f"No matching GT found for {len(missing_references)} result images: "
            f"{preview}. Use --result_suffix when result filenames contain a suffix."
        )
    if not pairs:
        raise ValueError(f"No image pairs found in {result_dir} and {reference_dir}.")

    return pairs[:ntest] if ntest is not None else pairs


def load_rgb_tensor(path, device):
    image = util_image.imread(path, chn="rgb", dtype="float32")
    return util_image.img2tensor(image).to(device)


def create_metrics(device):
    return {
        "psnr": pyiqa.create_metric(
            "psnr", test_y_channel=True, color_space="ycbcr"
        ).to(device),
        "ssim": pyiqa.create_metric(
            "ssim", test_y_channel=True, color_space="ycbcr"
        ).to(device),
        "lpips": pyiqa.create_metric("lpips").to(device),
    }


@torch.inference_mode()
def evaluate(result_dir, reference_dir, ntest=None, result_suffix="", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    pairs = build_image_pairs(
        result_dir,
        reference_dir,
        result_suffix=result_suffix,
        ntest=ntest,
    )
    metrics = create_metrics(device)
    totals = {name: 0.0 for name in METRIC_NAMES}

    print(f"Evaluating {len(pairs)} paired images on {device}...")
    for result_path, reference_path in tqdm(pairs, unit="image"):
        result_tensor = load_rgb_tensor(result_path, device)
        reference_tensor = load_rgb_tensor(reference_path, device)
        if result_tensor.shape != reference_tensor.shape:
            raise ValueError(
                f"Image shape mismatch: {result_path.name} {tuple(result_tensor.shape)} "
                f"vs {reference_path.name} {tuple(reference_tensor.shape)}"
            )

        for name, metric in metrics.items():
            totals[name] += metric(result_tensor, reference_tensor).item()

    averages = {name: value / len(pairs) for name, value in totals.items()}
    print("=" * 56)
    print(f"Images: {len(pairs)}")
    print(f"PSNR : {averages['psnr']:.4f}")
    print(f"SSIM : {averages['ssim']:.4f}")
    print(f"LPIPS: {averages['lpips']:.4f}")
    print("=" * 56)
    return averages


def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate Shift-IISR results with paired GT metrics."
    )
    parser.add_argument("--input", required=True, help="Folder containing results.")
    parser.add_argument("--reference", required=True, help="Folder containing GT images.")
    parser.add_argument("--ntest", type=int, default=None, help="Evaluate only the first N pairs.")
    parser.add_argument(
        "--result_suffix",
        default="",
        help="Suffix removed from result stems before matching GT, e.g. 'x4' or '_lr'.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Evaluation device, e.g. cuda, cuda:0, or cpu.",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    evaluate(
        args.input,
        args.reference,
        ntest=args.ntest,
        result_suffix=args.result_suffix,
        device=args.device,
    )
