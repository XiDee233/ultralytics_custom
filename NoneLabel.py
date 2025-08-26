import os
from pathlib import Path


def create_empty_labels(image_dir, label_dir=None, image_extensions=None):
    """
    为指定目录中的所有图像文件创建对应的空YOLO格式标签文件。

    参数:
        image_dir (str): 图像文件夹路径
        label_dir (str): 标签文件夹路径，默认为image_dir同级的"labels"文件夹
        image_extensions (list): 要处理的图像文件扩展名，默认为常见图像格式
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']

    # 确保扩展名列表中的所有元素都是小写且以点开头
    image_extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in image_extensions]

    # 设置标签目录（默认为同级目录下的"labels"文件夹）
    if label_dir is None:
        label_dir = Path(image_dir).parent / 'labels'

    # 创建标签目录
    Path(label_dir).mkdir(parents=True, exist_ok=True)

    # 遍历图像目录中的所有文件
    for root, _, files in os.walk(image_dir):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in image_extensions:
                # 获取图像文件的相对路径
                rel_path = os.path.relpath(root, image_dir)
                label_subdir = Path(label_dir) / rel_path

                # 创建子目录（如果不存在）
                label_subdir.mkdir(parents=True, exist_ok=True)

                # 构建对应的标签文件路径
                base_name = os.path.splitext(file)[0]
                txt_file = label_subdir / f"{base_name}.txt"

                # 创建空标签文件
                with open(txt_file, 'w') as f:
                    pass  # 空文件，不写入任何内容

                print(f"创建空标签: {txt_file}")

    print(f"\n已完成！所有空标签文件已创建在: {label_dir}")


if __name__ == "__main__":
    # 配置图像文件夹路径
    image_directory = r"D:\pyProject\ultralytics_custom\datasets\61fishs\dataset_20250603_014305_yolo\dataset\enhance\de"  # 替换为你的图像文件夹路径

    # 执行创建空标签操作
    create_empty_labels(image_directory,r"D:\pyProject\ultralytics_custom\datasets\61fishs\dataset_20250603_014305_yolo\dataset\enhance\de\label")