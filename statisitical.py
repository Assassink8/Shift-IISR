import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_ir_vis_distribution(vis_path, ir_path, save_dir="output"):

    os.makedirs(save_dir, exist_ok=True)

    # 读取图像
    vis = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

    if vis is None or ir is None:
        raise ValueError("Image loading failed")

    vis_pixels = vis.flatten()
    ir_pixels = ir.flatten()

    # seaborn风格
    sns.set(style="whitegrid")
    sns.set_context("paper", font_scale=1.0)


    # 图像比例 3.5:2
    fig_ratio = (3.5, 2)

    # ========================
    # 1 VIS 灰度分布
    # ========================
    plt.figure(figsize=fig_ratio)

    sns.histplot(
        vis_pixels,
        bins=100,
        kde=True,
        stat="density",
        color="#8ecae6"   # 淡蓝色
    )

    plt.xlabel("Gray Level")
    plt.ylabel("Density")
    plt.title("Visible Image Gray-Level Distribution")

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "vis_distribution.png"), dpi=300)
    plt.close()

    # ========================
    # 2 IR 灰度分布
    # ========================
    plt.figure(figsize=fig_ratio)

    sns.histplot(
        ir_pixels,
        bins=100,
        kde=True,
        stat="density",
        color="#ffb703"   # 淡橙色
    )

    plt.xlabel("Gray Level")
    plt.ylabel("Density")
    plt.title("Infrared Image Gray-Level Distribution")

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "ir_distribution.png"), dpi=300)
    plt.close()

    # ========================
    # 3 Overlay 对比
    # ========================
    plt.figure(figsize=fig_ratio)

    sns.kdeplot(
        vis_pixels,
        color="#8ecae6",
        label="Visible",
        linewidth=2
    )

    sns.kdeplot(
        ir_pixels,
        color="#ffb703",
        label="Infrared",
        linewidth=2
    )

    plt.xlabel("Gray Level")
    plt.ylabel("Density")
    plt.title("Gray-Level Distribution Comparison")

    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "overlay_distribution.png"), dpi=300)
    plt.close()

    print("Figures saved to:", save_dir)


if __name__ == "__main__":

    vis_path = "/share/huayunpeng-nfs/image_enhancement/ResShiftIR/testdata/data/data/test_set15/vis/03813.png"
    ir_path = "/share/huayunpeng-nfs/image_enhancement/ResShiftIR/testdata/data/data/test_set15/ir/03813.png"

    plot_ir_vis_distribution(vis_path, ir_path)