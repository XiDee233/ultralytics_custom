import os
import shutil
import numpy as np
from ultralytics import YOLO
def filter_bad_samples(
        model_weights,
        image_dir,
        label_dir,
        output_dir,
        conf_threshold=0.5,  # 低质量检测阈值
        include_fp=True,  # 是否包含误检样本
        include_fn=True,  # 是否包含漏检样本
        include_low_qual=True  # 是否包含低质量样本
):
    """
    筛选并输出漏检、误检或低质量的样本
    :param conf_threshold: 置信度阈值，低于此值视为低质量
    :param include_fp: 是否包含误检样本
    :param include_fn: 是否包含漏检样本
    :param include_low_qual: 是否包含低质量样本
    """
    # 准备目录
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    model = YOLO(model_weights)
    stats = {'total': 0, 'fp': 0, 'fn': 0, 'low_qual': 0, 'selected': 0}

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")

        # 读取真实标签
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                gt_boxes = [line.strip() for line in f if line.strip()]
        is_empty = len(gt_boxes) == 0

        # 模型预测
        results = model(img_path)
        pred_boxes = results[0].boxes if results else []
        confs = pred_boxes.conf.cpu().numpy() if pred_boxes else []

        # 初始化标志
        is_fp = False
        is_fn = False
        is_low_qual = False

        # 情况分析
        if is_empty:
            # 空照片但模型误检（False Positive）
            if len(pred_boxes) > 0 and include_fp:
                is_fp = True
                stats['fp'] += 1
        else:
            # 非空照片分析
            fn_count = max(0, len(gt_boxes) - len(pred_boxes))  # 漏检数
            fp_count = max(0,len(pred_boxes)-len(gt_boxes))
            low_qual = np.mean(confs) < conf_threshold if len(confs) > 0 else False  # 质量检测
            if fp_count > 0 and include_fp:
                is_fp = True
                stats['fp'] += 1
            if fn_count > 0 and include_fn:
                is_fn = True
                stats['fn'] += 1
            if low_qual and include_low_qual:
                is_low_qual = True
                stats['low_qual'] += 1

        # 判断是否需要输出
        if is_fp or is_fn or is_low_qual:
            stats['selected'] += 1
            # 复制图片
            shutil.copy(
                img_path,
                os.path.join(f"{output_dir}/images", img_name)
            )
            # 复制标签（空照片也复制空标签）
            dst_label = os.path.join(f"{output_dir}/labels", f"{base_name}.txt")
            if not os.path.exists(label_path):
                open(dst_label, 'w').close()  # 创建空标签文件
            else:
                shutil.copy(label_path, dst_label)

        stats['total'] += 1

    # 打印统计信息
    print(f"\n===== 筛选完成 =====")
    print(f"总处理图片: {stats['total']}")
    print(f"筛选出的样本: {stats['selected']}")
    print(f"误检样本(FPs): {stats['fp']}")
    print(f"漏检样本(FNs): {stats['fn']}")
    print(f"低质量样本: {stats['low_qual']}")
    print(f"筛选结果已保存至: {output_dir}")

def smart_copy_bad_samples(
        model_weights,
        image_dir,
        label_dir,
        output_dir,
        fp_weight=2,  # 误检惩罚权重
        fn_weight=3,  # 漏检惩罚权重
        qual_weight=1,  # 低质量惩罚权重
        max_copy=10  # 单张图片最大复制次数
):
    """
    智能复制差样本（包含空照片处理）
    :param fp_weight: 误检的复制权重（每误检1个目标复制N次）
    :param fn_weight: 漏检的复制权重（每漏检1个目标复制N次）
    :param qual_weight: 低质量检测的复制基数
    """
    # 准备目录
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    model = YOLO(model_weights)
    stats = {'total': 0, 'fp': 0, 'fn': 0, 'low_qual': 0}

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")

        # 读取真实标签
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                gt_boxes = [line.strip() for line in f if line.strip()]
        is_empty = len(gt_boxes) == 0

        # 模型预测
        results = model(img_path)
        pred_boxes = results[0].boxes if results else []
        confs = pred_boxes.conf.cpu().numpy() if pred_boxes else []

        # 情况分析
        copy_times = 1  # 默认至少复制1次

        if is_empty:
            # 空照片但模型误检（False Positive）
            if len(pred_boxes) > 0:
                fp_count = len(pred_boxes)
                copy_times += fp_count * fp_weight
                stats['fp'] += 1
        else:
            # 非空照片分析
            fn_count = max(0, len(gt_boxes) - len(pred_boxes))  # 漏检数
            low_qual = np.mean(confs) < 0.5 if len(confs) > 0 else False  # 质量检测
            fp_count = max(0,len(pred_boxes)-len(gt_boxes))

            if fn_count > 0 or low_qual or fp_count>0:
                copy_times += fn_count * fn_weight + (qual_weight if low_qual else 0)+fp_count*fp_weight
                if fn_count > 0: stats['fn'] += 1
                if low_qual: stats['low_qual'] += 1
                if fp_count>0:stats['fp']+=1

        # 限制最大复制次数
        copy_times = min(max_copy, copy_times)
        stats['total'] += 1

        # 执行复制
        for i in range(copy_times):
            suffix = f"_copy{i}" if i > 0 else ""
            # 复制图片
            shutil.copy(
                img_path,
                os.path.join(f"{output_dir}/images", f"{base_name}{suffix}.jpg")
            )
            # 复制标签（空照片也复制空标签）
            if os.path.exists(label_path) or is_empty:
                dst_label = os.path.join(f"{output_dir}/labels", f"{base_name}{suffix}.txt")
                if not os.path.exists(label_path):
                    open(dst_label, 'w').close()  # 创建空标签文件
                else:
                    shutil.copy(label_path, dst_label)

    # 打印统计信息
    print(f"\n===== 复制完成 =====")
    print(f"总处理图片: {stats['total']}")
    print(f"误检样本(FPs): {stats['fp']} (复制权重 x{fp_weight})")
    print(f"漏检样本(FNs): {stats['fn']} (复制权重 x{fn_weight})")
    print(f"低质量样本: {stats['low_qual']} (复制基数 +{qual_weight})")
    print(f"临时数据集已生成至: {output_dir}")

#
# # 使用示例
# filter_bad_samples(
#     model_weights=r'D:\Android\pyProj\ultralytics-main\ultralytics-main\runs\fish\detect\Archie\yolo11SEAM\train10\weights\best.pt',
#     image_dir=r'D:\Android\pyProj\ultralytics-main\ultralytics-main\datasets\61fishs\dataset_20250603_014305_yolo\dataset\enhance\split\train\images',
#     label_dir=r'D:\Android\pyProj\ultralytics-main\ultralytics-main\datasets\61fishs\dataset_20250603_014305_yolo\dataset\enhance\split\train\labels',
#     output_dir=r'D:\Android\pyProj\ultralytics-main\ultralytics-main\datasets\61fishs\dataset_20250603_014305_yolo\dataset\enhance\split\crash3',
#
# )
# 使用示例
smart_copy_bad_samples(
    model_weights=r'D:\Android\pyProj\ultralytics-main\ultralytics-main\runs\fish\detect\Archie\yolo11SEAM\train10\weights\best.pt',
    image_dir=r'D:\Android\pyProj\ultralytics-main\ultralytics-main\datasets\61fishs\dataset_20250603_014305_yolo\dataset\enhance\split\train\images',
    label_dir=r'D:\Android\pyProj\ultralytics-main\ultralytics-main\datasets\61fishs\dataset_20250603_014305_yolo\dataset\enhance\split\train\labels',
    output_dir=r'D:\Android\pyProj\ultralytics-main\ultralytics-main\datasets\61fishs\dataset_20250603_014305_yolo\dataset\enhance\splitNew',

)