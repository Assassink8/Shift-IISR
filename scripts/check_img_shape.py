import os
from PIL import Image

def check_images(folder_path):
    valid_ext = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
    count = 0
    ok_count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(valid_ext):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        
                        if width % 4 == 0 and height % 4 == 0:
                            # print(f"[OK] {file} -> {width}x{height}")
                            ok_count += 1
                            continue
                        else:
                            print(f"[NG] {file} -> {width}x{height} (不能被4整除)")
                            count += 1
                except Exception as e:
                    print(f"[ERROR] 无法处理 {file}: {e}")
    print(f"总共有 {count} 张图片的尺寸不能被4整除。")
    print(f"总共有 {ok_count} 张图片的尺寸可以被4整除。")

if __name__ == "__main__":
    folder = input("请输入要检查的文件夹路径: ").strip()
    check_images(folder)