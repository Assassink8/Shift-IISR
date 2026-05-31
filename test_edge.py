import torch
import kornia
import cv2
import numpy as np

def process_sobel_grayscale(input_path: str, output_path: str):
    # 1. 加载图像 (OpenCV 默认 BGR)
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        print(f"错误: 找不到文件 {input_path}")
        return

    # 2. 转换为 Tensor 并归一化到 [0, 1]
    # Kornia 转换: [H, W, C] -> [B, C, H, W]
    img_tensor = kornia.image_to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0) # 增加 Batch 维度

    # 3. 核心步骤：转换为灰度图
    # [1, 3, H, W] -> [1, 1, H, W]
    gray_tensor = kornia.color.rgb_to_grayscale(img_tensor)

    # 4. 计算 Sobel 边缘
    # kornia.filters.sobel 返回的是幅值图 (Magnitude)
    sobel_edges = kornia.filters.sobel(gray_tensor)

    # 5. 后处理与保存
    # 技巧：边缘图往往比较暗，可以进行一次简单的归一化增强对比度
    max_val = sobel_edges.max()
    if max_val > 0:
        sobel_edges = sobel_edges / max_val

    # 转回 NumPy: [1, 1, H, W] -> [H, W]
    edge_img = kornia.tensor_to_image(sobel_edges)
    
    # 转换为 8-bit 图像并保存
    edge_img_8bit = (edge_img * 255.0).astype(np.uint8)
    cv2.imwrite(output_path, edge_img_8bit)
    
    print(f"处理完成！灰度边缘图已保存至: {output_path}")

if __name__ == "__main__":
    process_sobel_grayscale("//share/huayunpeng-nfs/image_enhancement/ResShiftIR/testdata/data/data/test_set15/vis/03813.png", "vis_03813_edge.png")