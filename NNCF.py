#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 nncf.pdf 文档修正：移除代理依赖，解决下载失败问题
核心逻辑严格遵循 nncf.pdf 4-10 节（模型导出、量化、精度对比）
"""

from pathlib import Path
from functools import partial
from typing import Dict, Tuple

import torch
import openvino as ov
import nncf
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator as Validator
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.data.converter import coco80_to_coco91_class

# ========================= 1. 路径与参数（严格匹配 nncf.pdf 文档，使用本地文件）=========================
MODEL_PT = Path(r"D:\pyProject\ultralytics_custom\runs\fish\detect\Archie\yolo112025726\train13\weights\best.pt")
DATA_YAML = Path(
    r"D:\pyProject\ultralytics_custom\datasets\61fishs\dataset_20250603_014305_yolo\dataset\enhance\splitNew\data.yaml")
OUTPUT_DIR = Path(r"./out_yolo_int8")
TASK = "detect"
DEVICE = "CPU"
IMG_SIZE = 640
MAX_DROP = 0.01
DROP_TYPE = "ABSOLUTE"
SUBSET_SIZE = 128
RANKING_SUBSET_SIZE = 25
BATCH_SIZE = 1

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ========================= 2. 模型导出（nncf.pdf 4 节原文，无修改）=========================
def export_to_openvino(model_path: Path) -> Path:
    """nncf.pdf 4 节：PyTorch → OpenVINO IR，复用已导出文件"""
    model = YOLO(str(model_path))
    ir_dir = Path(f"{model_path.with_suffix('')}_openvino_model")

    # 复用已导出 IR，避免重复操作（nncf.pdf 隐含优化逻辑）
    if ir_dir.exists():
        ir_xml = next(ir_dir.glob("*.xml"))
        print(f"✅ 复用已导出 IR：{ir_xml}")
        return ir_xml

    # 按文档参数导出 IR（dynamic=True, half=False）
    model.export(format="openvino")
    ir_xml = next(ir_dir.glob("*.xml"))
    print(f"✅ 新导出 IR：{ir_xml}")
    return ir_xml


# ========================= 3. 验证函数（nncf.pdf 7 节原文，独立定义）=========================
def validation_ac(
        compiled_model: ov.CompiledModel,  # 第一参数：OpenVINO 编译模型（文档要求）
        validation_loader: torch.utils.data.DataLoader,  # 第二参数：数据集加载器（文档要求）
        validator: Validator,
        num_samples: int = None,
        log: bool = True,
) -> float:
    """nncf.pdf 7 节原文：计算 mAP50-95，用于量化精度控制"""
    validator.seen = 0
    validator.jdict = []
    # 按任务类型初始化 stats（贴合 nncf.pdf 文档任务适配逻辑）
    if TASK == "detect":
        # 检测任务：无'tp_m'（'tp_m'是分割任务掩码指标）
        validator.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    else:
        # 分割任务：保留'tp_m'（与 nncf.pdf 7 节原文一致）
        validator.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    validator.batch_i = 1
    validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)

    num_outputs = len(compiled_model.outputs)
    counter = 0

    for batch_i, batch in enumerate(validation_loader):
        if num_samples is not None and batch_i >= num_samples:
            break
        batch = validator.preprocess(batch)
        results = compiled_model(batch["img"])

        # 处理输出（nncf.pdf 7 节原文逻辑）
        if num_outputs == 1:
            preds = torch.from_numpy(results[compiled_model.output(0)])
        else:
            preds = [
                torch.from_numpy(results[compiled_model.output(0)]),
                torch.from_numpy(results[compiled_model.output(1)]),
            ]

        # 适配 Ultralytics 8.3.0 版本：临时添加 end2end 属性（nncf.pdf 文档隐含兼容需求）
        if not hasattr(validator, "end2end"):
            validator.end2end = False  # 检测任务无需 end2end 模式，设为 False 不影响逻辑

        preds = validator.postprocess(preds)  # 按 nncf.pdf 文档原文调用
        validator.update_metrics(preds, batch)
        counter += 1

    # ====================== 新增：NNCF 单样本兼容处理 ======================
    # 适配 nncf.pdf 8 节“多次验证调用”场景，处理单样本无结果导致的空列表
    if TASK == "detect":
        for k in validator.stats.keys():
            if k in ["tp", "conf", "pred_cls", "target_cls", "target_img"]:
                if not validator.stats[k] or all(t.numel() == 0 for t in validator.stats[k]):
                    validator.stats[k].append(torch.tensor([], dtype=torch.float32))
    # ======================================================================

    # 计算指标（nncf.pdf 7 节原文：直接索引，不使用 get()）
    stats = validator.get_stats()
    stats_metrics = stats["metrics/mAP50-95(B)"] if num_outputs == 1 else stats["metrics/mAP50-95(M)"]

    if log:
        print(f"Validate: dataset length = {counter}, metric value = {stats_metrics:.3f}")
    return stats_metrics


# ========================= 4. 验证器与数据加载器（nncf.pdf 5 节，使用本地数据）=========================
def get_validator_and_dataloader(model_path: Path, data_yaml: Path) -> Tuple[Validator, torch.utils.data.DataLoader]:
    """nncf.pdf 5 节：构建验证器，直接使用本地 data.yaml（无需下载）"""
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = str(data_yaml)  # 加载本地数据集配置（nncf.pdf 核心需求）
    args.imgsz = IMG_SIZE
    args.task = TASK
    args.mode = "val"
    args.batch = BATCH_SIZE

    model_ref = YOLO(str(model_path))
    validator_cls = model_ref.task_map[model_ref.task]["validator"]
    validator = validator_cls(args=args)

    # 验证器配置（nncf.pdf 5 节原文，含 COCO 类别映射）
    validator.data = check_det_dataset(args.data)  # 检查本地数据集格式
    validator.is_coco = True
    validator.class_map = coco80_to_coco91_class()
    validator.names = model_ref.model.names
    validator.metrics.names = validator.names
    validator.nc = model_ref.model.model[-1].nc

    # 修正：处理 stride 张量，避免布尔判断歧义（符合 nncf.pdf 动态获取逻辑）
    stride = getattr(getattr(model_ref, "model", None), "stride", None)
    if stride is not None:
        # 将张量转为 Python 列表（如 tensor([32.]) → [32]）
        stride_list = stride.tolist() if isinstance(stride, torch.Tensor) else stride
        validator.stride = int(stride_list[0]) if len(stride_list) > 0 else 32
    else:
        # 若无法获取，使用 nncf.pdf 文档隐含默认值 32
        validator.stride = 32

    # 分割任务配置（nncf.pdf 5 节原文）
    if TASK == "segment":
        validator.nm = 32
        validator.process = ops.process_mask
        validator.plot_masks = []

    # 创建本地数据加载器（nncf.pdf 5 节原文）
    dataloader = validator.get_dataloader(validator.data["val"], batch_size=args.batch)
    return validator, dataloader


# ========================= 5. NNCF 数据集封装（nncf.pdf 6 节原文）=========================
def create_nncf_dataset(validator: Validator, dataloader: torch.utils.data.DataLoader) -> nncf.Dataset:
    """nncf.pdf 6 节：将本地数据加载器转为 NNCF 兼容格式"""

    def transform_fn(data_item: Dict):
        preprocessed = validator.preprocess(data_item)
        return preprocessed["img"].numpy()

    return nncf.Dataset(dataloader, transform_fn)

import time  # 确保开头导入 time 模块

def evaluate_performance(
        compiled_model: ov.CompiledModel,
        validation_loader: torch.utils.data.DataLoader,
        validator: Validator,
        log: bool = True,
) -> tuple[float, float]:  # 主流程用：返回（精度，平均推理时间）
    """扩展 nncf.pdf 10 节：评估精度+推理时间，供主流程性能对比"""
    # 1. 复用 validation_ac 的精度计算逻辑
    validator.seen = 0
    validator.jdict = []
    if TASK == "detect":
        validator.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    else:
        validator.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
    validator.batch_i = 1
    validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)

    num_outputs = len(compiled_model.outputs)
    counter = 0
    total_infer_time = 0.0  # 新增：统计总推理时间
    total_images = 0        # 新增：统计总图片数（处理批次）

    for batch_i, batch in enumerate(validation_loader):
        batch = validator.preprocess(batch)
        # 2. 统计推理时间（仅含模型推理，排除预处理）
        infer_start = time.time()
        results = compiled_model(batch["img"])
        infer_end = time.time()
        batch_infer_time = infer_end - infer_start
        total_infer_time += batch_infer_time
        total_images += batch["img"].shape[0]  # 批次大小：[B, C, H, W] 中的 B

        # 3. 后续精度计算逻辑（与 validation_ac 一致）
        if num_outputs == 1:
            preds = torch.from_numpy(results[compiled_model.output(0)])
        else:
            preds = [
                torch.from_numpy(results[compiled_model.output(0)]),
                torch.from_numpy(results[compiled_model.output(1)]),
            ]

        if not hasattr(validator, "end2end"):
            validator.end2end = False
        preds = validator.postprocess(preds)
        validator.update_metrics(preds, batch)
        counter += 1

    # 4. 兼容空列表（与 validation_ac 一致）
    if TASK == "detect":
        for k in validator.stats.keys():
            if k in ["tp", "conf", "pred_cls", "target_cls", "target_img"]:
                if not validator.stats[k] or all(t.numel() == 0 for t in validator.stats[k]):
                    validator.stats[k].append(torch.tensor([], dtype=torch.float32))

    # 5. 计算精度+平均时间
    stats = validator.get_stats()
    stats_metrics = stats["metrics/mAP50-95(B)"] if num_outputs == 1 else stats["metrics/mAP50-95(M)"]
    avg_infer_time = total_infer_time / total_images if total_images > 0 else 0.0

    # 6. 打印结果（主流程用）
    if log:
        print(f"\n性能评估：")
        print(f"  验证集批次：{counter}")
        print(f"  总图片数：{total_images}")
        print(f"  mAP50-95：{stats_metrics:.4f}")
        print(f"  平均推理时间：{avg_infer_time:.4f} 秒/张")
        print(f"  FPS：{1/avg_infer_time:.2f} 帧/秒")

    return stats_metrics, avg_infer_time  # 主流程用：返回精度+时间
# ========================= 6. 主流程（nncf.pdf 8-10 节，移除代理相关代码）=========================
def main():
    # Step 1：导出模型（nncf.pdf 4 节）
    ir_xml = export_to_openvino(MODEL_PT)

    # Step 2：构建验证器与本地数据加载器（nncf.pdf 5 节，无代理请求）
    validator, dataloader = get_validator_and_dataloader(MODEL_PT, DATA_YAML)

    # Step 3：封装 NNCF 数据集（nncf.pdf 6 节）
    nncf_dataset = create_nncf_dataset(validator, dataloader)

    # Step 4：创建验证函数（nncf.pdf 7 节：partial 绑定参数）
    validation_fn = partial(validation_ac, validator=validator, log=False)

    # Step 5：加载原始模型（nncf.pdf 8 节）
    core = ov.Core()
    original_model = core.read_model(str(ir_xml))

    # Step 6：带精度控制的量化（nncf.pdf 8 节原文参数）
    # Step 6：NNCF 量化（用纯精度的 validation_ac，符合 NNCF 要求）
    print("\n🚀 开始量化（nncf.pdf 8 节流程）...")
    # 绑定 NNCF 所需的验证函数（仅返回精度，与 validation_ac 一致）
    validation_fn = partial(validation_ac, validator=validator, log=False)
    quantized_model = nncf.quantize_with_accuracy_control(
        model=original_model,
        calibration_dataset=nncf_dataset,
        validation_dataset=nncf_dataset,
        validation_fn=validation_fn,  # 关键：NNCF 调用纯精度验证
        preset=nncf.QuantizationPreset.MIXED,
        max_drop=MAX_DROP,
        subset_size=SUBSET_SIZE,
        drop_type=nncf.DropType[DROP_TYPE],
        advanced_accuracy_restorer_parameters=AdvancedAccuracyRestorerParameters(
            ranking_subset_size=RANKING_SUBSET_SIZE
        ),
    )

    # Step 7：保存量化模型（不变）
    quant_xml = OUTPUT_DIR / f"{Path(ir_xml).stem}_int8.xml"
    ov.save_model(quantized_model, str(quant_xml), compress_to_fp16=False)
    print(f"\n✅ 量化模型保存：{quant_xml}")

    # Step 8：主流程性能对比（用新函数 evaluate_performance，含精度+时间）
    print("\n" + "=" * 80)
    print("📊 精度+推理时间对比（扩展 nncf.pdf 10 节流程）")
    print("=" * 80)
    core = ov.Core()
    compiled_original = core.compile_model(original_model, DEVICE)
    compiled_quant = core.compile_model(quantized_model, DEVICE)

    # 调用新函数：评估原始模型和量化模型的精度+时间
    original_mAP, original_avg_time = evaluate_performance(compiled_original, dataloader, validator)
    quant_mAP, quant_avg_time = evaluate_performance(compiled_quant, dataloader, validator)

    # 计算关键指标
    mAP_drop = original_mAP - quant_mAP
    time_speedup = original_avg_time / quant_avg_time if original_avg_time > 0 else 0.0  # 加速比

    # 输出对比结果
    print("\n" + "=" * 80)
    print(f"{'指标':<20} {'原始模型':<15} {'量化模型':<15} {'变化':<10}")
    print("-" * 80)
    print(f"mAP50-95        {original_mAP:.4f}        {quant_mAP:.4f}        {(-mAP_drop / original_mAP * 100):+.2f}%")
    print(
        f"平均推理时间(秒) {original_avg_time:.4f}        {quant_avg_time:.4f}        {((quant_avg_time - original_avg_time) / original_avg_time * 100):+.2f}%")
    print(
        f"FPS(帧/秒)      {(1 / original_avg_time):.2f}        {(1 / quant_avg_time):.2f}        {((1 / quant_avg_time - 1 / original_avg_time) / (1 / original_avg_time) * 100):+.2f}%")
    print(f"推理加速比      {'1.00x':<15}        {time_speedup:.2f}x        {((time_speedup - 1) * 100):+.2f}%")
    print("-" * 80)
    print(f"精度下降（绝对）：{mAP_drop:.4f}（允许最大值：{MAX_DROP}）")
    print("=" * 80)


if __name__ == "__main__":
    main()