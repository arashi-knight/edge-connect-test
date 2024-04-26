import os
import shutil

from support.config import Config

def copy_images(source_path, destination_path):
    # 检查目标路径是否存在，如果不存在则创建
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # 遍历目标路径下的所有文件和文件夹
    for root, dirs, files in os.walk(source_path):
        for file in files:
            # 检查文件是否是图片文件
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # 构建源文件路径和目标文件路径
                source_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, source_path)
                destination_folder = os.path.join(destination_path, relative_path)
                destination_file_path = os.path.join(destination_folder, file)

                # 检查目标文件夹是否存在，如果不存在则创建
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)

                # 复制文件
                shutil.copyfile(source_file_path, destination_file_path)
                print('复制文件:', source_file_path, '到', destination_file_path)

# 保存log
def save_log(config: Config):

    source_path = r'E:\CODE\project\manga\comic-inpaint\results\1'
    destination_path = 'output'
    copy_images(source_path, destination_path)
    pass