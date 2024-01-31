import os
import shutil
import re

def extract_images(source_folder, destination_folder):
    # 获取源文件夹中所有图片文件的路径
    image_files = [file for file in os.listdir(source_folder) if file.endswith('.png')]

    # 按文件名后缀为.jpg的浮点数进行排序
    image_files.sort(key=lambda x: float(re.findall(r'\d+\.\d+', x)[0]))

    # 创建目标文件夹
    os.makedirs(destination_folder, exist_ok=True)

    # 每隔三张图片中的一张进行复制
    for i in range(0, len(image_files), 4):
        image_file = image_files[i]
        source_path = os.path.join(source_folder, image_file)
        destination_path = os.path.join(destination_folder, image_file)
        shutil.copy2(source_path, destination_path)

# 指定源文件夹路径和目标文件夹路径
source_folder = '/home/air/multinerf/dataset/distributed_nerf/NewYork/2/rgb'
destination_folder = '/home/air/multinerf/dataset/distributed_nerf/NewYork/2/images'

# 调用函数提取图片
extract_images(source_folder, destination_folder)