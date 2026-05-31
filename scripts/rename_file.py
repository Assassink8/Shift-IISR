# import os

# folder = "/share/huayunpeng-local/raw/M3FD/M3FD_Fusion/result"

# files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])

# # 第一步：改成临时文件名（避免覆盖）
# for filename in files:
#     old_path = os.path.join(folder, filename)
#     temp_path = os.path.join(folder, "temp_" + filename)
#     os.rename(old_path, temp_path)

# # 第二步：改成最终名字（数字+1）
# temp_files = sorted([f for f in os.listdir(folder) if f.startswith("temp_")])

# for filename in temp_files:
#     old_path = os.path.join(folder, filename)

#     # 提取原始数字
#     num = int(filename.replace("temp_", "").replace(".png", ""))
    
#     new_name = f"{num + 1}.png"
#     new_path = os.path.join(folder, new_name)

#     os.rename(old_path, new_path)

# print("重命名完成！")


import os

folder = "/share/huayunpeng-nfs/image_enhancement/CoRPLE/results/my_test_CoRPLE_light_x4_M3FD_Detection/visualization/M3FD_Detection"

for filename in os.listdir(folder):
    if filename.endswith("_lr_hr.png"):
        num = int(filename.replace("_lr_hr.png", ""))
        new_name = f"{num + 1}.png"
        
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)
        
        if os.path.exists(new_path):
            print(f"⚠️ 跳过: {new_name} 已存在")
            continue
        
        print(f"{filename} -> {new_name}")
        os.rename(old_path, new_path)