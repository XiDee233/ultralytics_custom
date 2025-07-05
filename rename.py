import os


def batch_rename_files(directory):
    """
    对指定目录下以d开头的文件添加a_前缀

    参数:
        directory (str): 要处理的目录路径
    """
    # 确保目录存在
    if not os.path.exists(directory):
        print(f"错误: 指定的目录 '{directory}' 不存在")
        return

    try:
        # 获取目录中的所有文件
        files = os.listdir(directory)

        # 过滤以d开头的文件
        d_files = [f for f in files if os.path.isfile(os.path.join(directory, f)) and f.startswith('a_')]

        if not d_files:
            print("在指定目录中没有找到以'd'开头的文件")
            return

        # 重命名文件
        for filename in d_files:
            old_path = os.path.join(directory, filename)
            new_filename = f"a_{filename}"
            new_path = os.path.join(directory, new_filename)

            # 重命名文件
            os.rename(old_path, new_path)
            print(f"已重命名: {filename} -> {new_filename}")

        print(f"成功重命名 {len(d_files)} 个文件")

    except Exception as e:
        print(f"发生错误: {e}")

batch_rename_files(r"C:\Users\16652\Pictures\fishfry67")
