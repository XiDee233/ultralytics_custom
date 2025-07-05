import os
import random
import shutil

import cv2
import numpy as np
import torch
import torchvision.models as models
import pnnx
from sklearn.model_selection import KFold

from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')  # 设置后端为 Agg


def get_random_color(seed=None):
    """
    生成鲜明的随机颜色

    参数:
        seed: 随机数种子，若指定则生成固定颜色

    返回:
        BGR格式的颜色元组 (B, G, R)
    """
    if seed is not None:
        random.seed(seed)

    # 生成HSV空间中的颜色，确保亮度和饱和度足够高，以获得鲜明的颜色
    hue = random.randint(0, 180)  # HSV中的色调范围是0-180
    saturation = random.randint(100, 255)  # 饱和度范围
    value = random.randint(100, 255)  # 亮度范围

    # 将HSV颜色转换为BGR格式
    hsv_color = np.array([[[hue, saturation, value]]], dtype=np.uint8)
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)

    return tuple(int(c) for c in bgr_color[0][0])
if __name__ == '__main__':
    x = input("请输入执行的操作，")
    if x == 'y' or x == 'Y':
        # yolo = YOLO(r'ultralytics/cfg/models/11/yolo11n.yaml')
        # 先加载权重
        yolo = YOLO(r'D:\Android\pyProj\ultralytics-main\ultralytics-main\runs\fish\detect\Archie\yolo11\train35\weights\best.pt')
        # yolo.train(data=r'mydataset.yaml',
        #            workers=8, epochs=300, batch=16, resume=True, project='runs/detect/Archie')

        yolo.train(data=r'datasets/SuLinSheng/data.yaml',
                   project='runs/fish/detect/Archie/yolo11',
                   device='0',
                   epochs=30,  # 训练过程中整个数据集将被迭代多少次,显卡不行你就调小点
                   batch=20,  # 一次看完多少张图片才进行权重更新
                   workers=4,
                   imgsz=640,
                   lr0=0.00005
                   )
        # print(yolo.overrides)  # 查看所有默认参数
    elif x == 'xx':
        # 数据集路径
        data_dir = 'datasets/syb2'
        image_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')

        # 获取所有图像和标签文件
        image_files = sorted(os.listdir(image_dir))
        label_files = sorted(os.listdir(label_dir))

        # 使用现有模型对所有图片进行预测
        initial_model = YOLO(r'D:\Android\pyProj\ultralytics-main\ultralytics-main\runs\fish\detect\Archie\yolo11_fold_4\train\weights\best.pt')

        # 计算每张图片的置信度得分
        image_scores = {}
        print("正在评估图片...")
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)
            results = initial_model(img_path)
            confidences = [box.conf.item() for box in results[0].boxes]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            image_scores[img_file] = avg_conf
            print(f"{img_file}: 置信度 {avg_conf:.3f}")

        # 计算重复次数（置信度越低，重复次数越多）
        repeat_counts = {}
        max_repeats = 4  # 最大重复次数
        min_conf = min(image_scores.values())
        max_conf = max(image_scores.values())
        conf_range = max_conf - min_conf

        for img_file, score in image_scores.items():
            if conf_range > 0:
                # 将置信度映射到重复次数（1-3次）
                repeat = max(1, int(1 + (max_repeats - 1) * (max_conf - score) / conf_range))
            else:
                repeat = 1
            repeat_counts[img_file] = repeat
            print(f"{img_file} 将被重复 {repeat} 次")

        # 定义 K 折交叉验证
        k = 5
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # 循环进行 K 折交叉验证
        for fold, (train_index, val_index) in enumerate(kf.split(image_files)):
            print(f"Fold {fold + 1}/{k}")

            # 创建临时数据集目录
            temp_data_dir = f'datasets/temp_fold_{fold}'
            try:
                os.makedirs(temp_data_dir, exist_ok=True)
                os.makedirs(os.path.join(temp_data_dir, 'train', 'images'), exist_ok=True)
                os.makedirs(os.path.join(temp_data_dir, 'train', 'labels'), exist_ok=True)
                os.makedirs(os.path.join(temp_data_dir, 'val', 'images'), exist_ok=True)
                os.makedirs(os.path.join(temp_data_dir, 'val', 'labels'), exist_ok=True)
            except OSError as e:
                print(f"创建目录失败: {e}")
                continue

            # 修改训练集复制逻辑
            try:
                for i in train_index:
                    img_file = image_files[i]
                    label_file = label_files[i]
                    repeats = repeat_counts[img_file]

                    # 复制图片和标签指定的次数
                    for r in range(repeats):
                        if r == 0:  # 第一次使用原始文件名
                            dst_img = os.path.join(temp_data_dir, 'train', 'images', img_file)
                            dst_label = os.path.join(temp_data_dir, 'train', 'labels', label_file)
                        else:  # 之后的复制使用带后缀的文件名
                            base_name, ext = os.path.splitext(img_file)
                            dst_img = os.path.join(temp_data_dir, 'train', 'images', f"{base_name}_r{r}{ext}")
                            base_name, ext = os.path.splitext(label_file)
                            dst_label = os.path.join(temp_data_dir, 'train', 'labels', f"{base_name}_r{r}{ext}")

                        shutil.copy(os.path.join(image_dir, img_file), dst_img)
                        shutil.copy(os.path.join(label_dir, label_file), dst_label)

                # 验证集保持不变
                for i in val_index:
                    shutil.copy(os.path.join(image_dir, image_files[i]),
                              os.path.join(temp_data_dir, 'val', 'images', image_files[i]))
                    shutil.copy(os.path.join(label_dir, label_files[i]),
                              os.path.join(temp_data_dir, 'val', 'labels', label_files[i]))
            except shutil.Error as e:
                print(f"复制文件失败: {e}")
                continue

            # 创建临时数据配置文件
            temp_data_yaml = f'{temp_data_dir}/data.yaml'
            train_path = os.path.abspath(os.path.join(temp_data_dir, 'train', 'images'))
            val_path = os.path.abspath(os.path.join(temp_data_dir, 'val', 'images'))

            data_config = {
                "train": train_path,
                "val": val_path,
                "nc": 1,
                "names": ["fry"],
                # 数据增强相关配置
                "augment": True,
                "hsv_h": 0.4,
                "hsv_s": 0.4,
                "hsv_v": 0.4,
                "fliplr": 0.6,
                "mosaic": 1.0,
                "mixup": 0.4,
                # 其他常见数据增强参数
                "flipud": 0.0,  # 上下翻转概率
                "rotate": 10.0,  # 旋转角度范围（度）
                "translate": 0.1,  # 平移比例
                "scale": 0.5,  # 缩放比例
                "shear": 2.0,  # 剪切角度（度）
                "perspective": 0.0,  # 透视变换强度
                "copy_paste": 0.0,  # 复制粘贴增强概率
                "cutout": 0.0,  # CutOut增强概率
                "grid_mix": 0.0  # GridMix增强概率
            }

            try:
                with open(temp_data_yaml, 'w') as f:
                    for key, value in data_config.items():
                        if isinstance(value, list):
                            value_str = ", ".join([str(v) for v in value])
                            f.write(f"{key}: [{value_str}]\n")
                        else:
                            f.write(f"{key}: {value}\n")
            except IOError as e:
                print(f"写入数据配置文件失败: {e}")

            # 初始化模型（第一轮使用初始模型，后续使用上一轮的模型）
            try:
                if fold == 0:
                    print("第一轮训练，加载初始模型...")
                    # 重新加载模型而不是直接使用initial_model
                    yolo = YOLO(r'D:\Android\pyProj\ultralytics-main\ultralytics-main\runs\fish\detect\Archie\yolo11_fold_4\train\weights\best.pt')
                    # 打印一些模型信息
                    print(f"模型类型: {type(yolo.model)}")
                    print(f"模型权重示例: {next(yolo.model.parameters())[:5]}")  # 打印部分权重值
                else:
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    model_path = f'runs/fish/detect/Archie/syb2/yolo11_fold_{fold-1}/train/weights/last.pt'
                    yolo = YOLO(model_path)
            except Exception as e:
                print(f"加载模型失败: {e}")
                continue

            # 训练模型
            try:
                # 设置CUDA内存分配器配置
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

                yolo.train(data=temp_data_yaml,
                          project=f'runs/fish/detect/Archie/syb2/yolo11_fold_{fold}',
                          device='0',
                          epochs=30,
                          batch=20,  # 减小批量大小
                          imgsz=640,  # 减小图像尺寸
                          workers=2,  # 减少工作进程数
                          lr0=0.00005,
                          cache=False,  # 禁用缓存
                          nbs=8,  # 标称批量大小
                          )  # 梯度累积步数
            except Exception as e:
                print(f"训练模型失败: {e}")
                continue

            # 删除临时数据集目录
            try:
                shutil.rmtree(temp_data_dir)
            except OSError as e:
                print(f"删除临时数据集目录失败: {e}")
    elif x == 'c':

        model = YOLO(
            r'D:\Android\pyProj\ultralytics-main\ultralytics-main\runs\fish\detect\Archie\yolo11\train5\weights\best.pt')  # load an official model


        model.export(format="ncnn", imgsz=320, opset=12, optimize=False,dynamic=False)
        # model = models.resnet18(pretrained=True)
        #
        # # 2. 创建一个虚拟输入（YOLOv8 使用 640x640 输入）
        # x = torch.rand(1, 3, 640, 640)
        #
        # opt_model =pnnx.export(model, r'D:/Android/pyProj/ultralytics-main/ultralytics-main/runs/detect/Archie/train13/weights/best.pt', x)
        # result = opt_model(x)
    elif x == 'nn':
        model = YOLO(
            r'D:\Android\pyProj\ultralytics-main\ultralytics-main\runs\fish\detect\Archie\yolo11NoSeam\train\weights\best.pt')

        # 图像路径
        image_path = r"C:\Users\16652\Pictures\fishfry67\a_a_dframe_0350.jpg"

        # 读取图像
        frame = cv2.imread(image_path)

        # 确保图像已成功加载
        if frame is None:
            print(f"错误：无法加载图像 {image_path}")
        else:
            # 模型预测
            results = model.predict(frame,
                                    # conf=0.75,  # 提高置信度阈值，减少低置信度检测
                                    # iou=0.6,  # 降低IoU阈值，减少重叠框
                                    # agnostic_nms=True  # 类别无关NMS，适合单类检测
                                    )

            # 获取检测结果
            boxes = results[0].boxes

            # 使用OpenCV绘制更细的边界框
            annotated_frame = frame.copy()
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 获取类别和置信度
                cls = int(box.cls)
                conf = float(box.conf)

                # 使用之前定义的随机颜色函数
                color = get_random_color()  # 使用类别ID作为种子，确保同类目标颜色一致

                # 绘制边界框（线宽为1像素）
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # 绘制标签
                label = f"{results[0].names[cls]} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

            # 创建可调整大小的窗口
            cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)

            # 设置窗口大小（例如：宽度800像素，高度600像素）
            cv2.resizeWindow("YOLOv8 Detection", 1920, 1080)

            # 显示结果
            cv2.imshow("YOLOv8 Detection", annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif x == 'n':
        # 加载导出的 NCNN 模型
        model = YOLO(r'D:\Android\pyProj\ultralytics-main\ultralytics-main\runs\fish\detect\Archie\yolo11FRFN\train2\weights\best.pt')
        # 打开视频文件
        video_path = r"C:\Users\16652\Documents\WeChat Files\wxid_2il9e2877tum22\FileStorage\Video\2025-05\017ae61abed77200ee0820c84fcc60a7_raw.mp4"
        cap = cv2.VideoCapture(video_path)

        # 获取原始视频分辨率
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 设置显示分辨率（例如缩小为原来的50%）
        scale_percent = 50
        display_width = int(original_width * scale_percent / 100)
        display_height = int(original_height * scale_percent / 100)
        display_size = (display_width, display_height)

        # 创建可调整大小的窗口
        cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv8 Detection", display_width, display_height)

        # 检查视频是否成功打开
        if not cap.isOpened():
            print("Error opening video file")
            exit()

        # 初始化暂停状态
        paused = False

        while True:
            # 如果没有暂停，则读取下一帧
            if not paused:
                ret, frame = cap.read()

                # 如果视频结束，退出循环
                if not ret:
                    break

                # 调整帧大小以降低分辨率
                frame = cv2.resize(frame, display_size)

                # 进行目标检测
                results = model.predict(frame,
                                        conf = 0.5,
                                        iou=0.6,  # 降低IoU阈值，减少重叠框
                                        )

                # 获取检测结果
                boxes = results[0].boxes

                # 使用OpenCV绘制更细的边界框
                annotated_frame = frame.copy()
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 获取类别和置信度
                    cls = int(box.cls)
                    conf = float(box.conf)
                    color = get_random_color()
                    # 绘制边界框（线宽为1像素）
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)

                    # 绘制标签
                    label = f"{results[0].names[cls]} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 显示当前状态
                status_text = "Press SPACE to pause"
                cv2.putText(annotated_frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 显示结果
                cv2.imshow("YOLOv8 Detection", annotated_frame)
            else:
                # 暂停状态下显示提示
                status_text = "Paused - Press SPACE to continue"
                cv2.putText(annotated_frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("YOLOv8 Detection", annotated_frame)

            # 处理键盘事件
            key = cv2.waitKey(1) & 0xFF

            # 按空格键切换暂停状态
            if key == 32:  # 32 是空格键的 ASCII 码
                paused = not paused

            # 按 'q' 键退出循环
            elif key == ord('q'):
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()




    elif x == 'seam':
        model = YOLO(r"D:\Android\pyProj\ultralytics-main\ultralytics-main\ultralytics\cfg\models\11\yolo11_FRFN.yaml").\
            load(r"D:\Android\pyProj\ultralytics-main\ultralytics-main\runs\fish\yolo11n.pt")
        # 直接加载上次训练的模型（包含训练状态）
        # model = YOLO(
            # r"D:\Android\pyProj\ultralytics-main\ultralytics-main\runs\fish\detect\Archie\yolo11Syb\train3\weights\last.pt")
        # model = YOLO(r"D:\Android\pyProj\ultralytics-main\ultralytics-main\ultralytics\cfg\models\11\yolo11.yaml").\
        #     load(r"D:\Android\pyProj\ultralytics-main\ultralytics-main\runs\fish\yolo11n.pt")
        results = model.train(data=r'D:\Android\pyProj\ultralytics-main\ultralytics-main\datasets\61fishs\dataset_20250603_014305_yolo\dataset\enhance\data.yaml',
                   project='runs/fish/detect/Archie/yolo11FRFN',
                   device='0',
                   epochs=60,
                   batch=16,  # 一次看完多少张图片才进行权重更新
                   workers=4,
                   imgsz=640,
                   lr0=0.0001)