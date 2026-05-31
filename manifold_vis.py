import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import umap
import plotly.express as px
import plotly.graph_objects as go


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class TwoFolderImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"读取图片失败: {path}, 错误: {e}")

        img = self.transform(img)
        return img, label, str(path), Path(path).name


def collect_images(folder1, folder2, label1="folder1", label2="folder2"):
    image_paths = []
    labels = []

    for p in sorted(Path(folder1).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            image_paths.append(p)
            labels.append(label1)

    for p in sorted(Path(folder2).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            image_paths.append(p)
            labels.append(label2)

    return image_paths, labels


def build_feature_extractor(device):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])  # 去掉最后分类层
    model.eval().to(device)
    preprocess = weights.transforms()
    return model, preprocess


@torch.no_grad()
def extract_features(image_paths, labels, batch_size=32, num_workers=4, device="cpu"):
    model, preprocess = build_feature_extractor(device)
    dataset = TwoFolderImageDataset(image_paths, labels, preprocess)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda"))
    )

    all_features = []
    all_labels = []
    all_paths = []
    all_names = []

    for imgs, batch_labels, batch_paths, batch_names in tqdm(loader, desc="提取特征"):
        imgs = imgs.to(device, non_blocking=True)
        feats = model(imgs)              # [B, 2048, 1, 1]
        feats = feats.flatten(1)         # [B, 2048]
        feats = feats.cpu().numpy()

        all_features.append(feats)
        all_labels.extend(batch_labels)
        all_paths.extend(batch_paths)
        all_names.extend(batch_names)

    all_features = np.concatenate(all_features, axis=0)
    return all_features, all_labels, all_paths, all_names


def compute_umap_3d(features, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42):
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    embedding = reducer.fit_transform(features)
    return embedding


def build_dataframe(embedding, labels, paths, names):
    df = pd.DataFrame({
        "x": embedding[:, 0],
        "y": embedding[:, 1],
        "z": embedding[:, 2],
        "label": labels,
        "path": paths,
        "name": names,
    })
    return df


def find_name_pairs(df, label1, label2):
    """
    找两个文件夹中同名文件的配对。
    仅在同一个 name 下，恰好同时包含 label1 和 label2 时连线。
    如果某个名字在某一侧重复出现多个文件，这里只取第一个。
    """
    pairs = []

    for name, group in df.groupby("name"):
        g1 = group[group["label"] == label1]
        g2 = group[group["label"] == label2]

        if len(g1) >= 1 and len(g2) >= 1:
            p1 = g1.iloc[0]
            p2 = g2.iloc[0]
            pairs.append((p1, p2))

    return pairs


def add_pair_lines(fig, pairs):
    for p1, p2 in pairs:
        dist = float(np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2))

        fig.add_trace(go.Scatter3d(
            x=[p1.x, p2.x],
            y=[p1.y, p2.y],
            z=[p1.z, p2.z],
            mode="lines",
            line=dict(width=2, color="gray"),
            showlegend=False,
            hoverinfo="text",
            text=[f"{p1.name}<br>distance={dist:.4f}", f"{p2.name}<br>distance={dist:.4f}"],
        ))


def make_figure(df, pairs, title="3D Image Manifold"):
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="label",
        hover_name="name",
        hover_data=["path"],
        title=title,
    )

    fig.update_traces(marker=dict(size=4))
    add_pair_lines(fig, pairs)

    fig.update_layout(
        legend_title_text="Folder",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="两个文件夹图片的三维流形可视化，并连接同名文件")
    parser.add_argument("--folder1", type=str, required=True, help="第一个图片文件夹")
    parser.add_argument("--folder2", type=str, required=True, help="第二个图片文件夹")
    parser.add_argument("--label1", type=str, default="folder1", help="第一个文件夹显示名称")
    parser.add_argument("--label2", type=str, default="folder2", help="第二个文件夹显示名称")
    parser.add_argument("--output_html", type=str, default="image_manifold_3d_pairs.html", help="输出HTML文件")
    parser.add_argument("--output_csv", type=str, default="image_manifold_3d_pairs.csv", help="输出CSV文件")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker数")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--metric", type=str, default="cosine", help="UMAP metric")
    args = parser.parse_args()

    print("收集图片中...")
    image_paths, labels = collect_images(
        args.folder1, args.folder2,
        label1=args.label1, label2=args.label2
    )

    if len(image_paths) == 0:
        raise ValueError("没有找到任何图片，请检查文件夹路径。")

    print(f"共找到 {len(image_paths)} 张图片")
    print(f"{args.label1}: {sum(1 for x in labels if x == args.label1)} 张")
    print(f"{args.label2}: {sum(1 for x in labels if x == args.label2)} 张")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    features, labels, paths, names = extract_features(
        image_paths=image_paths,
        labels=labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device
    )

    print("UMAP 降维到三维中...")
    embedding = compute_umap_3d(
        features,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric
    )

    df = build_dataframe(embedding, labels, paths, names)
    pairs = find_name_pairs(df, args.label1, args.label2)

    print(f"找到可连线的同名文件对: {len(pairs)} 对")

    # 保存 CSV
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"坐标已保存到: {args.output_csv}")

    # 绘图
    fig = make_figure(
        df,
        pairs,
        title=f"3D Image Manifold: {args.label1} vs {args.label2}"
    )
    fig.write_html(args.output_html)
    print(f"交互式图已保存到: {args.output_html}")


if __name__ == "__main__":
    main()