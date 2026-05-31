#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper-analysis utility for comparing GT IR, ResShift, DifIISR, and Ours
distributions in the ResShift VQModelTorch/VQGAN autoencoder latent space.

This script reuses the repository autoencoder config, model construction, image
loading, and checkpoint loading utilities. It extracts encoder latents, pools
them into one vector per image, then saves features plus t-SNE/UMAP plots and
coordinates for reproducible paper figures.
"""

import argparse
import random
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import util_image, util_net  # noqa: E402
from utils import util_common  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASS_SPECS = (
    ("GT IR", "GT_IR", "gt_dir"),
    ("ResShift", "ResShift", "resshift_dir"),
    ("DifIISR", "DifIISR", "difiisr_dir"),
    ("Ours", "Ours", "ours_dir"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize VQModelTorch latent distributions with t-SNE/UMAP."
    )
    parser.add_argument("--gt_dir", type=str, required=True, help="GT infrared image folder.")
    parser.add_argument("--resshift_dir", type=str, required=True, help="ResShift output folder.")
    parser.add_argument("--difiisr_dir", type=str, required=True, help="DifIISR output folder.")
    parser.add_argument("--ours_dir", type=str, required=True, help="Ours output folder.")
    parser.add_argument("--config", type=str, required=True, help="ResShift config path.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional autoencoder checkpoint. Overrides config autoencoder ckpt_path.",
    )
    parser.add_argument("--save_dir", type=str, default="./vis_vq_latent", help="Output directory.")
    parser.add_argument("--max_samples", type=int, default=500, help="Max images per class.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resize", type=int, default=None, help="Resize images to S x S if provided.")
    parser.add_argument(
        "--crop_multiple",
        type=int,
        default=64,
        help="Center crop H/W to this multiple when --resize is not set.",
    )
    parser.add_argument(
        "--pool",
        type=str,
        default="gap",
        choices=["gap", "flatten"],
        help="Latent pooling mode: global average pooling or flatten.",
    )
    parser.add_argument("--run_tsne", action="store_true", help="Generate t-SNE visualization.")
    parser.add_argument("--run_umap", action="store_true", help="Generate UMAP visualization.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity.")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    args = parser.parse_args()

    if not args.run_tsne and not args.run_umap:
        args.run_tsne = True
        args.run_umap = True
    return args


def resolve_path(path_value, config_path=None, must_exist=False):
    if path_value is None:
        return None
    path = Path(str(path_value)).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(REPO_ROOT / path)
        if config_path is not None:
            candidates.append(Path(config_path).resolve().parent / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    resolved = candidates[0].resolve()
    if must_exist:
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def get_config_section(configs, section_names):
    for name in section_names:
        if name in configs and configs.get(name) is not None:
            return name, configs.get(name)
    return None, None


def checkpoint_to_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "params", "model", "autoencoder", "first_stage_model"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
    return ckpt


def strip_prefix_once(key, prefixes):
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def flexible_load_state_dict(model, state):
    prefixes = (
        "module.",
        "_orig_mod.",
        "autoencoder.",
        "first_stage_model.",
        "first_stage_model.module.",
        "model.",
        "base_ae.",
    )
    model_state = model.state_dict()
    converted = OrderedDict()

    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        candidates = [key, strip_prefix_once(key, prefixes)]
        for candidate in candidates:
            if candidate in model_state and model_state[candidate].shape == value.shape:
                converted[candidate] = value
                break

    if not converted:
        raise RuntimeError("No checkpoint tensors matched the autoencoder state_dict.")

    missing, unexpected = model.load_state_dict(converted, strict=False)
    loaded = len(converted)
    total = len(model_state)
    print(f"Loaded {loaded}/{total} autoencoder tensors with flexible prefix matching.")
    if missing:
        print(f"Missing keys after flexible load: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys after flexible load: {len(unexpected)}")
    return loaded, total


def load_checkpoint(model, ckpt_path, device):
    print(f"Loading AutoEncoder model from {ckpt_path}...")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = checkpoint_to_state_dict(ckpt)

    try:
        util_net.reload_model(model, state)
        print("Loaded autoencoder with utils.util_net.reload_model.")
    except Exception as exc:
        print(f"utils.util_net.reload_model failed: {exc}")
        flexible_load_state_dict(model, state)


def instantiate_autoencoder(configs, config_path, ckpt_override, device):
    section_name, ae_config = get_config_section(
        configs, ("autoencoder", "first_stage_model", "first_stage_config")
    )
    if ae_config is None:
        raise KeyError(
            "Could not find autoencoder/first_stage_model/first_stage_config in config."
        )
    if "target" not in ae_config:
        raise KeyError(f"Config section '{section_name}' has no target field.")

    params = ae_config.get("params", {})
    model = util_common.get_obj_from_str(ae_config.target)(**params)
    model = model.to(device)

    ckpt_path = ckpt_override if ckpt_override is not None else ae_config.get("ckpt_path", None)
    if ckpt_path is not None:
        ckpt_path = resolve_path(ckpt_path, config_path=config_path, must_exist=True)
        load_checkpoint(model, ckpt_path, device)
    else:
        print(f"No ckpt_path found for config section '{section_name}'. Using initialized weights.")

    model = model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    print(f"Using config section '{section_name}' target: {ae_config.target}")
    return model, section_name


def list_images(image_dir, max_samples, seed):
    image_dir = resolve_path(image_dir, must_exist=True)
    paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    if not paths:
        raise FileNotFoundError(f"No supported images found in {image_dir}")

    rng = random.Random(seed)
    rng.shuffle(paths)
    if max_samples is not None and max_samples > 0:
        paths = paths[:max_samples]
    return paths


def center_crop_to_multiple(image, multiple, path):
    if multiple is None or multiple <= 1:
        return image
    height, width = image.shape[:2]
    crop_h = height - height % multiple
    crop_w = width - width % multiple
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(
            f"{path} has size {height}x{width}, smaller than crop_multiple={multiple}. "
            "Use --resize or a smaller --crop_multiple."
        )
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return image[top:top + crop_h, left:left + crop_w, :]


def load_image_tensor(path, resize, crop_multiple):
    image = util_image.imread(
        path,
        chn="rgb",
        dtype="float32",
        force_gray2rgb=True,
        force_rgba2rgb=True,
    )
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image after RGB conversion, got {image.shape}: {path}")
    if image.shape[2] > 3:
        image = image[:, :, :3]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] != 3:
        raise ValueError(f"Expected 3 image channels, got {image.shape}: {path}")

    if resize is not None:
        image = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_AREA)
    else:
        image = center_crop_to_multiple(image, crop_multiple, path)

    image = np.ascontiguousarray(image)
    tensor = util_image.img2tensor(image).squeeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor


class ImageFolderDataset(Dataset):
    def __init__(self, paths, resize=None, crop_multiple=64):
        self.paths = paths
        self.resize = resize
        self.crop_multiple = crop_multiple

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = load_image_tensor(path, self.resize, self.crop_multiple)
        return str(path), tensor


def collate_as_list(batch):
    paths, tensors = zip(*batch)
    return list(paths), list(tensors)


def unwrap_encode_output(encoded):
    if isinstance(encoded, (tuple, list)):
        encoded = encoded[0]
    if torch.is_tensor(encoded):
        return encoded
    if hasattr(encoded, "mode"):
        value = encoded.mode()
        if torch.is_tensor(value):
            return value
    if hasattr(encoded, "sample"):
        value = encoded.sample()
        if torch.is_tensor(value):
            return value
    raise TypeError(f"Unsupported model.encode output type: {type(encoded)}")


def pool_latent(z, pool):
    if pool == "gap" and z.dim() == 4:
        return z.mean(dim=(2, 3))
    return z.flatten(1)


def encode_tensor_batch(model, tensors, device, pool):
    x = torch.stack(tensors, dim=0).to(device, non_blocking=True)
    encoded = model.encode(x)
    z = unwrap_encode_output(encoded)
    feat = pool_latent(z, pool)
    return feat.detach().float().cpu().numpy()


def extract_features_for_class(model, paths, args, device, label):
    dataset = ImageFolderDataset(paths, resize=args.resize, crop_multiple=args.crop_multiple)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_as_list,
    )
    features = []
    ordered_paths = []

    with torch.no_grad():
        for batch_paths, tensors in tqdm(dataloader, desc=f"Encoding {label}", leave=False):
            by_shape = {}
            for path, tensor in zip(batch_paths, tensors):
                by_shape.setdefault(tuple(tensor.shape), ([], []))
                by_shape[tuple(tensor.shape)][0].append(path)
                by_shape[tuple(tensor.shape)][1].append(tensor)

            for _, (shape_paths, shape_tensors) in by_shape.items():
                features.append(encode_tensor_batch(model, shape_tensors, device, args.pool))
                ordered_paths.extend(shape_paths)

    try:
        features = np.concatenate(features, axis=0)
    except ValueError as exc:
        raise ValueError(
            "Could not concatenate latent features. If --pool flatten is used, all images "
            "must produce the same latent shape; pass --resize to enforce this."
        ) from exc

    return features, ordered_paths


def save_center_distances(features_by_label, save_dir):
    gt_center = features_by_label["GT IR"].mean(axis=0)
    lines = ["Center distance to GT IR:"]
    print("\nCenter distance to GT IR:")
    for label in ("ResShift", "DifIISR", "Ours"):
        center = features_by_label[label].mean(axis=0)
        distance = float(np.linalg.norm(center - gt_center))
        line = f"{label}: {distance:.6f}"
        lines.append(line)
        print(line)

    path = save_dir / "center_distance_to_gt.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def make_pair_key(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        rel_path = path.relative_to(root)
    except ValueError:
        rel_path = Path(path.name)

    stem = rel_path.stem
    if stem.endswith("_lr"):
        stem = stem[:-3]
    return (rel_path.parent / stem).as_posix()


def build_feature_lookup(features, paths, root, label):
    lookup = {}
    duplicate_keys = []
    for feature, path in zip(features, paths):
        key = make_pair_key(path, root)
        if key in lookup:
            duplicate_keys.append(key)
        lookup[key] = feature

    if duplicate_keys:
        print(
            f"Warning: {label} has {len(duplicate_keys)} duplicate pair keys. "
            "Later samples overwrote earlier samples."
        )
    return lookup


def save_paired_feature_distances(features_by_label, paths_by_label, roots_by_label, save_dir):
    gt_lookup = build_feature_lookup(
        features_by_label["GT IR"],
        paths_by_label["GT IR"],
        roots_by_label["GT IR"],
        "GT IR",
    )

    lines = ["Paired feature distance to GT IR:"]
    print("\nPaired feature distance to GT IR:")

    for label in ("ResShift", "DifIISR", "Ours"):
        sr_lookup = build_feature_lookup(
            features_by_label[label],
            paths_by_label[label],
            roots_by_label[label],
            label,
        )
        matched_keys = sorted(set(sr_lookup.keys()) & set(gt_lookup.keys()))
        missing_in_gt = sorted(set(sr_lookup.keys()) - set(gt_lookup.keys()))
        missing_in_sr = sorted(set(gt_lookup.keys()) - set(sr_lookup.keys()))

        if not matched_keys:
            line = (
                f"{label}: nan "
                f"(matched=0, missing_gt={len(missing_in_gt)}, missing_sr={len(missing_in_sr)})"
            )
            lines.append(line)
            print(line)
            continue

        distances = np.array(
            [np.linalg.norm(sr_lookup[key] - gt_lookup[key]) for key in matched_keys],
            dtype=np.float32,
        )
        mean_distance = float(distances.mean())
        np.save(save_dir / f"{label}_paired_feature_distance_to_gt.npy", distances)

        line = (
            f"{label}: {mean_distance:.6f} "
            f"(matched={len(matched_keys)}, "
            f"missing_gt={len(missing_in_gt)}, missing_sr={len(missing_in_sr)})"
        )
        lines.append(line)
        print(line)

    path = save_dir / "paired_feature_distance_to_gt.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def plot_2d(coords, labels, title, save_path):
    markers = {
        "GT IR": "o",
        "ResShift": "s",
        "DifIISR": "^",
        "Ours": "D",
    }
    plt.figure(figsize=(7, 6), dpi=160)
    for label in ("GT IR", "ResShift", "DifIISR", "Ours"):
        mask = labels == label
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            marker=markers[label],
            alpha=0.78,
            linewidths=0.0,
            label=label,
        )
    ax = plt.gca()
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def run_tsne(features, labels, args, save_dir):
    if features.shape[0] < 3:
        raise ValueError("t-SNE needs at least 3 total samples.")
    perplexity = min(args.perplexity, max(1.0, features.shape[0] - 1.0))
    if perplexity != args.perplexity:
        print(f"Adjusted t-SNE perplexity from {args.perplexity} to {perplexity}.")
    coords = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
        perplexity=perplexity,
    ).fit_transform(features)
    np.save(save_dir / "vq_latent_tsne_coords.npy", coords)
    fig_path = save_dir / "vq_latent_tsne.png"
    plot_2d(coords, labels, "VQ latent t-SNE", fig_path)
    print(f"Saved t-SNE figure to {fig_path}")


def run_umap(features, labels, args, save_dir):
    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP requested but umap-learn is not installed: pip install umap-learn") from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
    )
    coords = reducer.fit_transform(features)
    np.save(save_dir / "vq_latent_umap_coords.npy", coords)
    fig_path = save_dir / "vq_latent_umap.png"
    plot_2d(coords, labels, "VQ latent UMAP", fig_path)
    print(f"Saved UMAP figure to {fig_path}")


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config_path = resolve_path(args.config, must_exist=True)
    configs = OmegaConf.load(config_path)
    if isinstance(configs, DictConfig):
        OmegaConf.resolve(configs)

    model, section_name = instantiate_autoencoder(configs, config_path, args.ckpt, device)

    features_by_label = OrderedDict()
    paths_by_label = OrderedDict()
    roots_by_label = OrderedDict()
    all_paths = []
    for label, file_prefix, arg_name in CLASS_SPECS:
        roots_by_label[label] = resolve_path(getattr(args, arg_name), must_exist=True)
        paths = list_images(roots_by_label[label], args.max_samples, args.seed)
        features, ordered_paths = extract_features_for_class(model, paths, args, device, label)
        features_by_label[label] = features
        paths_by_label[label] = ordered_paths
        all_paths.extend((label, path) for path in ordered_paths)
        np.save(save_dir / f"{file_prefix}_vq_latent.npy", features)
        print(f"{label} feature shape: {list(features.shape)}")

    save_center_distances(features_by_label, save_dir)
    save_paired_feature_distances(features_by_label, paths_by_label, roots_by_label, save_dir)

    features = np.concatenate(list(features_by_label.values()), axis=0)
    labels = np.concatenate(
        [np.full(class_features.shape[0], label, dtype="<U8")
         for label, class_features in features_by_label.items()]
    )
    np.save(save_dir / "vq_latent_labels.npy", labels)

    with (save_dir / "vq_latent_paths.txt").open("w") as handle:
        for label, path in all_paths:
            handle.write(f"{label}\t{path}\n")

    scaled_features = StandardScaler().fit_transform(features)
    if args.run_tsne:
        run_tsne(scaled_features, labels, args, save_dir)
    if args.run_umap:
        run_umap(scaled_features, labels, args, save_dir)

    print(f"Saved latent features and labels to {save_dir}")
    print(f"Autoencoder loaded from config section: {section_name}")


if __name__ == "__main__":
    main()
