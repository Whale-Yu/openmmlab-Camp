import os

source_dir = 'image/train'  # 源文件夹路径
k = 1
for filename in os.listdir(source_dir):
    if "jpg" in filename:
        old_path = os.path.join(source_dir, filename)  # 原始文件路径
        new_path = os.path.join(source_dir, "face_" + str(k) + ".jpg")  # 新文件路径
        os.rename(old_path, new_path)  # 重命名文件
        k += 1
