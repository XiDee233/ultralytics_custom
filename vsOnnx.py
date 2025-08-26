#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推理时间对比脚本（异步流水线版）
对比对象：ONNX 模型 | 原始 OpenVINO 模型 | 量化 OpenVINO 模型
核心指标：平均推理时间（秒/张）、FPS（帧/秒）、总推理时间
"""

import time
import csv
import cv2
import torch
import onnxruntime as ort
import openvino as ov
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
from threading import Lock
import random


# ========================= 1. 配置参数（需根据用户本地路径修改）=========================
class Config:
    # 模型路径（用户需替换为自己的模型路径）
    ONNX_MODEL_PATH = Path(
        r"D:\pyProject\ultralytics_custom\runs\fish\detect\Archie\yolo112025726\train13\weights\best.onnx")
    ORIGINAL_OV_MODEL_PATH = Path(
        r"D:\pyProject\ultralytics_custom\runs\fish\detect\Archie\yolo112025726\train13\weights\best_openvino_model\best.xml")
    QUANTIZED_OV_MODEL_PATH = Path(r"D:\pyProject\ultralytics_custom\out_yolo_int8\best_int8.xml")

    # 视频输入配置
    VIDEO_INPUT = r"D:\repos\QtWidgetsApplication1\x64\Debug\fish720.mp4"
    # VIDEO_INPUT = 0  # 摄像头输入

    # 模型与推理配置
    MODEL_INPUT_SIZE = (640, 640)  # (W, H)
    INPUT_BATCH = 1
    INPUT_CHANNELS = 3
    DEVICE = "CPU"  # OpenVINO设备
    ONNX_INPUT_NAME = "images"
    SAVE_CSV = True
    CSV_SAVE_PATH = Path("./async_video_inference_stats.csv")
    USE_ASYNC_PIPELINE = True  # 是否启用异步流水线
    SHOW_VIDEO = False  # 是否显示视频和检测框


# ========================= 2. 视频预处理工具 =========================
def preprocess_video_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[float, float]]:
    """视频帧预处理：缩放、归一化、格式转换"""
    original_h, original_w = frame.shape[:2]
    target_w, target_h = target_size
    x_scale = original_w / target_w
    y_scale = original_h / target_h

    # 保持比例缩放并填充黑边
    scale = min(target_w / original_w, target_h / original_h)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    padded_frame = cv2.copyMakeBorder(
        resized_frame, pad_h, target_h - new_h - pad_h, pad_w, target_w - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # 格式转换与归一化
    rgb_frame = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
    chw_frame = np.transpose(rgb_frame, (2, 0, 1))
    batch_frame = np.expand_dims(chw_frame, axis=0)
    input_tensor = torch.tensor(batch_frame, dtype=torch.float32) / 255.0

    return input_tensor, (x_scale, y_scale)


# ========================= 3. 后处理和可视化函数 =========================
def postprocess_yolo_output(output: np.ndarray, conf_threshold: float = 0.5, iou_threshold: float = 0.4) -> Tuple[List, List, List]:
    """YOLO输出后处理：NMS + 置信度过滤"""
    # YOLO输出格式: [batch, 84, 8400] -> [batch, 8400, 84]
    if len(output.shape) == 3:
        output = output.transpose(0, 2, 1)  # [1, 8400, 84]
    
    predictions = output[0]  # [8400, 84]
    
    # 提取边界框和置信度
    boxes = predictions[:, :4]  # [x_center, y_center, width, height]
    scores = predictions[:, 4:].max(axis=1)  # 最大类别置信度
    class_ids = predictions[:, 4:].argmax(axis=1)  # 类别ID
    
    # 置信度过滤
    valid_indices = scores > conf_threshold
    boxes = boxes[valid_indices]
    scores = scores[valid_indices]
    class_ids = class_ids[valid_indices]
    
    if len(boxes) == 0:
        return [], [], []
    
    # 转换为xyxy格式
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    xyxy_boxes = np.column_stack([x1, y1, x2, y2])
    
    # NMS
    indices = cv2.dnn.NMSBoxes(xyxy_boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    
    if len(indices) > 0:
        indices = indices.flatten()
        final_boxes = xyxy_boxes[indices]
        final_scores = scores[indices]
        final_class_ids = class_ids[indices]
        return final_boxes.tolist(), final_scores.tolist(), final_class_ids.tolist()
    
    return [], [], []


def draw_detections(frame: np.ndarray, boxes: List, scores: List, class_ids: List, 
                   class_names: List[str] = None, scale_factors: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """在帧上绘制检测结果"""
    if not boxes:
        return frame
    
    # 生成随机颜色
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(100)]
    
    x_scale, y_scale = scale_factors
    
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = box
        
        # 缩放回原始图像尺寸
        x1 = int(x1 * x_scale)
        y1 = int(y1 * y_scale)
        x2 = int(x2 * x_scale)
        y2 = int(y2 * y_scale)
        
        # 绘制边界框
        color = colors[class_id % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {score:.2f}"
        else:
            label = f"Class {class_id}: {score:.2f}"
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame


# ========================= 4. 异步推理基类 =========================
import functools


class AsyncInferPipeline:
    def __init__(self, model_path: Path, device: str, model_type: str):
        self.model_path = model_path
        self.device = device
        self.model_type = model_type
        self.core = ov.Core()
        self.model = self.core.read_model(str(model_path))
        self.compiled_model = self.core.compile_model(self.model, device)

        # 使用两个独立的InferRequest，模仿C++实现
        self.curr_request = self.compiled_model.create_infer_request()
        self.next_request = self.compiled_model.create_infer_request()

        self.callback_lock = Lock()
        self.infer_results = [None, None]
        self.infer_times = [0.0, 0.0]
        self.start_times = [0.0, 0.0]
        self.completed_flags = [False, False]  # 用于跟踪推理完成状态
        self.inference_outputs = [None, None]  # 保存推理输出结果

        # 设置回调
        self._setup_callbacks()

        # 预热推理
        self._warmup()

    def _setup_callbacks(self):
        """为每个InferRequest设置回调函数"""
        # 为curr_request设置回调 (request_id = 0)
        def curr_callback(userdata):
            end_time = time.time()
            with self.callback_lock:
                self.infer_times[0] = end_time - self.start_times[0]
                self.infer_results[0] = self.curr_request.get_output_tensor()
                self.inference_outputs[0] = self.curr_request.get_output_tensor().data.copy()
                self.completed_flags[0] = True
        
        # 为next_request设置回调 (request_id = 1)
        def next_callback(userdata):
            end_time = time.time()
            with self.callback_lock:
                self.infer_times[1] = end_time - self.start_times[1]
                self.infer_results[1] = self.next_request.get_output_tensor()
                self.inference_outputs[1] = self.next_request.get_output_tensor().data.copy()
                self.completed_flags[1] = True
        
        # 设置回调函数
        self.curr_request.set_callback(curr_callback, 0)
        self.next_request.set_callback(next_callback, 1)

    def _warmup(self):
        """预热推理以排除初始化耗时"""
        dummy_input = torch.randn(
            1, Config.INPUT_CHANNELS,
            Config.MODEL_INPUT_SIZE[1],
            Config.MODEL_INPUT_SIZE[0]
        )
        input_np = dummy_input.numpy()
        ov_tensor = ov.Tensor(input_np)
        
        # 预热两个请求
        self.curr_request.set_input_tensor(ov_tensor)
        self.curr_request.infer()
        
        self.next_request.set_input_tensor(ov_tensor)
        self.next_request.infer()

    def start_async_infer(self, request_id: int, input_tensor: torch.Tensor) -> None:
        """启动异步推理"""
        input_np = input_tensor.numpy()
        ov_tensor = ov.Tensor(input_np)
        
        with self.callback_lock:
            self.completed_flags[request_id] = False
            self.start_times[request_id] = time.time()
        
        if request_id == 0:
            self.curr_request.set_input_tensor(ov_tensor)
            self.curr_request.start_async()
        else:
            self.next_request.set_input_tensor(ov_tensor)
            self.next_request.start_async()

    def wait_for_result(self, request_id: int) -> float:
        """等待特定请求的推理结果并返回耗时"""
        if request_id == 0:
            self.curr_request.wait()
        else:
            self.next_request.wait()
        
        with self.callback_lock:
            return self.infer_times[request_id]
    
    def get_request_by_id(self, request_id: int):
        """根据ID获取对应的请求对象"""
        return self.curr_request if request_id == 0 else self.next_request
    
    def get_inference_output(self, request_id: int):
        """获取推理输出结果"""
        with self.callback_lock:
            return self.inference_outputs[request_id]


# ========================= 4. 性能统计与基准测试 =========================
def benchmark_model_with_video(
        model_type: str,
        video_cap: cv2.VideoCapture,
        infer_func: callable,
        is_async: bool = False,
        show_video: bool = True
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """用视频测试单个模型的推理性能"""
    frame_id = 0
    total_infer_time = 0.0
    total_process_time = 0.0
    frame_stats = []

    # 重置视频指针
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"\n📹 开始用视频测试 {model_type} 模型...")
    start_total_time = time.time()

    # 异步推理特定变量
    if is_async and model_type != "ONNX":  # ONNX不支持异步流水线
        current_req_id = 0
        next_req_id = 1
        # 先处理第一帧作为初始帧
        ret, frame = video_cap.read()
        if not ret:
            return {}, []

        input_tensor, _ = preprocess_video_frame(frame, Config.MODEL_INPUT_SIZE)
        infer_func.start_async_infer(current_req_id, input_tensor)
        frame_id += 1

    # 逐帧处理视频
    while True:
        # 读取视频帧
        ret, frame = video_cap.read()
        if not ret:
            # 处理最后一帧
            if is_async and model_type != "ONNX" and frame_id > 0:
                infer_time = infer_func.wait_for_result(current_req_id)
                total_infer_time += infer_time
            break

        frame_start_time = time.time()
        frame_id += 1

        # 推理处理
        if is_async and model_type != "ONNX":
            # 异步流水线处理 - 模仿C++的真正并行处理
            # 1. 预处理当前帧（与前一帧推理并行）
            input_tensor, scale_factors = preprocess_video_frame(frame, Config.MODEL_INPUT_SIZE)
            preprocess_time = time.time() - frame_start_time
            
            # 2. 启动当前帧的推理
            infer_func.start_async_infer(next_req_id, input_tensor)
            
            # 3. 等待前一帧推理完成（此时当前帧已经在并行处理）
            infer_time = infer_func.wait_for_result(current_req_id)
            total_infer_time += infer_time
            
            # 4. 获取推理结果并进行后处理
            if frame_id > 1 and show_video:  # 跳过第一帧，因为没有前一帧结果
                output = infer_func.get_inference_output(current_req_id)
                if output is not None:
                    boxes, scores, class_ids = postprocess_yolo_output(output)
                    frame_with_detections = draw_detections(frame, boxes, scores, class_ids, scale_factors=scale_factors)
                    cv2.imshow(f"{model_type} Detection", frame_with_detections)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print(f"\n用户按下 'q' 键，提前退出 {model_type} 测试")
                        break
            elif show_video:
                # 显示原始帧（没有检测结果时）
                cv2.imshow(f"{model_type} Video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"\n用户按下 'q' 键，提前退出 {model_type} 测试")
                    break
            
            # 5. 交换请求ID
            current_req_id, next_req_id = next_req_id, current_req_id
        else:
            # 同步处理
            input_tensor, scale_factors = preprocess_video_frame(frame, Config.MODEL_INPUT_SIZE)
            preprocess_time = time.time() - frame_start_time
            infer_time = infer_func(input_tensor)
            total_infer_time += infer_time
            
            # 显示检测结果（ONNX模型暂时跳过后处理显示）
            if show_video and model_type != "ONNX":
                cv2.imshow(f"{model_type} Video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # 计算处理时间
        frame_total_time = time.time() - frame_start_time
        total_process_time += frame_total_time

        # 记录统计信息
        frame_stats.append({
            "model_type": model_type,
            "frame_id": frame_id,
            "preprocess_time": round(preprocess_time * 1000, 3),
            "infer_time": round(infer_time * 1000, 3),
            "frame_total_time": round(frame_total_time * 1000, 3),
            "frame_fps": round(1 / frame_total_time, 2) if frame_total_time > 0 else 0.0
        })

        # 打印进度
        if frame_id % 10 == 0:
            current_fps = frame_id / (time.time() - start_total_time)
            print(f"  {model_type}：已处理 {frame_id} 帧 | 当前 FPS：{current_fps:.2f}")

    # 计算总性能统计
    total_frames = frame_id
    avg_infer_time = total_infer_time / total_frames if total_frames > 0 else 0.0
    avg_process_time = total_process_time / total_frames if total_frames > 0 else 0.0
    total_fps = total_frames / (time.time() - start_total_time) if (time.time() - start_total_time) > 0 else 0.0

    total_stats = {
        "model_type": model_type,
        "total_frames": total_frames,
        "total_process_time": round(time.time() - start_total_time, 4),
        "avg_infer_time_ms": round(avg_infer_time * 1000, 3),
        "avg_process_time_ms": round(avg_process_time * 1000, 3),
        "total_fps": round(total_fps, 2)
    }

    # 打印结果
    print(f"\n✅ {model_type} 视频推理完成：")
    print(f"  - 总帧数：{total_stats['total_frames']}")
    print(f"  - 总处理时间：{total_stats['total_process_time']:.2f} 秒")
    print(f"  - 平均推理耗时：{total_stats['avg_infer_time_ms']:.3f} 毫秒/帧")
    print(f"  - 平均总处理耗时：{total_stats['avg_process_time_ms']:.3f} 毫秒/帧")
    print(f"  - 视频处理 FPS：{total_stats['total_fps']:.2f} 帧/秒")

    return total_stats, frame_stats


# ========================= 5. 推理函数实现 =========================
def create_onnx_infer_func(model_path: Path, input_name: str, device: str) -> callable:
    """创建ONNX模型推理函数"""
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX模型不存在：{model_path}")

    providers = ["CPUExecutionProvider"]
    if device.lower() in ["gpu", "cuda"] and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    session = ort.InferenceSession(str(model_path), providers=providers)
    print(f"🔧 ONNX Runtime初始化完成，使用设备：{providers[0]}")

    # 预热推理
    dummy_input = torch.randn(1, Config.INPUT_CHANNELS, Config.MODEL_INPUT_SIZE[1], Config.MODEL_INPUT_SIZE[0])
    session.run(None, {input_name: dummy_input.numpy()})

    def onnx_infer(input_tensor: torch.Tensor) -> float:
        input_np = input_tensor.numpy()
        start = time.time()
        session.run(None, {input_name: input_np})
        return time.time() - start

    return onnx_infer


def create_async_ov_infer_pipeline(model_path: Path, device: str, model_type: str) -> AsyncInferPipeline:
    """创建OpenVINO异步推理流水线"""
    if not model_path.exists():
        raise FileNotFoundError(f"OpenVINO模型不存在：{model_path}")

    return AsyncInferPipeline(model_path, device, model_type)


# ========================= 6. 结果保存与可视化 =========================
def save_stats_to_csv(all_frame_stats: List[Dict[str, float]], save_path: Path):
    """保存帧级统计到CSV"""
    if not all_frame_stats:
        print("⚠️ 无帧级统计数据，不保存CSV")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_frame_stats[0].keys())
        writer.writeheader()
        writer.writerows(all_frame_stats)

    print(f"\n💾 帧级统计已保存到：{save_path}")


def print_final_comparison(all_total_stats: List[Dict[str, float]]):
    """打印性能对比表格"""
    print("\n" + "=" * 120)
    print(
        f"{'模型类型':<25} {'总帧数':<10} {'总处理时间(秒)':<20} {'平均推理耗时(毫秒)':<25} {'视频处理FPS(帧/秒)':<20}")
    print("-" * 120)

    for stats in all_total_stats:
        print(
            f"{stats['model_type']:<25} "
            f"{stats['total_frames']:<10} "
            f"{stats['total_process_time']:<20.2f} "
            f"{stats['avg_infer_time_ms']:<25.3f} "
            f"{stats['total_fps']:<20.2f}"
        )

    print("-" * 120)
    # 计算性能提升
    onnx_stats = next(s for s in all_total_stats if s["model_type"] == "ONNX")
    ov_original_stats = next(s for s in all_total_stats if s["model_type"] == "Original_OpenVINO")
    ov_quant_stats = next(s for s in all_total_stats if s["model_type"] == "Quantized_OpenVINO")

    ov_vs_onnx_fps_gain = ((ov_original_stats["total_fps"] - onnx_stats["total_fps"]) / onnx_stats["total_fps"]) * 100
    quant_vs_onnx_fps_gain = ((ov_quant_stats["total_fps"] - onnx_stats["total_fps"]) / onnx_stats["total_fps"]) * 100
    quant_vs_original_fps_gain = ((ov_quant_stats["total_fps"] - ov_original_stats["total_fps"]) / ov_original_stats[
        "total_fps"]) * 100

    print(f"📈 性能提升对比（以ONNX为基准）：")
    print(f"  - 原始OpenVINO比ONNX FPS提升：{ov_vs_onnx_fps_gain:+.2f}%")
    print(f"  - 量化OpenVINO比ONNX FPS提升：{quant_vs_onnx_fps_gain:+.2f}%")
    print(f"  - 量化OpenVINO比原始OpenVINO FPS提升：{quant_vs_original_fps_gain:+.2f}%")
    print("=" * 120)


# ========================= 7. 主流程 =========================
def main():
    print("=" * 120)
    print("📊 异步流水线视频推理性能对比（ONNX vs OpenVINO vs 量化OpenVINO）")
    print("=" * 120)
    print(f"配置信息：")
    print(f"  视频输入：{Config.VIDEO_INPUT}")
    print(f"  模型输入尺寸：{Config.MODEL_INPUT_SIZE[0]}x{Config.MODEL_INPUT_SIZE[1]}")
    print(f"  OpenVINO设备：{Config.DEVICE}")
    print(f"  启用异步流水线：{Config.USE_ASYNC_PIPELINE}")
    print(f"  保存帧级统计：{Config.SAVE_CSV}（路径：{Config.CSV_SAVE_PATH}）")
    print("=" * 120)


    video_cap = cv2.VideoCapture(Config.VIDEO_INPUT)
    if not video_cap.isOpened():
        raise ValueError(f"无法打开视频输入：{Config.VIDEO_INPUT}")
    print(
        f"\n✅ 视频捕获初始化完成，视频分辨率：{int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")



    onnx_infer = create_onnx_infer_func(Config.ONNX_MODEL_PATH, Config.ONNX_INPUT_NAME, Config.DEVICE)

    # OpenVINO使用异步流水线
    ov_original_pipeline = create_async_ov_infer_pipeline(
        Config.ORIGINAL_OV_MODEL_PATH, Config.DEVICE, "Original_OpenVINO")

    ov_quant_pipeline = create_async_ov_infer_pipeline(
        Config.QUANTIZED_OV_MODEL_PATH, Config.DEVICE, "Quantized_OpenVINO")


    # 性能测试
    all_total_stats = []
    all_frame_stats = []

    # # 测试ONNX模型（同步）
    # onnx_total, onnx_frame = benchmark_model_with_video(
    #     "ONNX", video_cap, onnx_infer, is_async=False, show_video=Config.SHOW_VIDEO)
    # all_total_stats.append(onnx_total)
    # all_frame_stats.extend(onnx_frame)

    # 测试原始OpenVINO模型（异步）
    ov_original_total, ov_original_frame = benchmark_model_with_video(
        "Original_OpenVINO", video_cap, ov_original_pipeline, is_async=Config.USE_ASYNC_PIPELINE, show_video=Config.SHOW_VIDEO)
    all_total_stats.append(ov_original_total)
    all_frame_stats.extend(ov_original_frame)

    # 测试量化OpenVINO模型（异步）
    ov_quant_total, ov_quant_frame = benchmark_model_with_video(
        "Quantized_OpenVINO", video_cap, ov_quant_pipeline, is_async=Config.USE_ASYNC_PIPELINE, show_video=Config.SHOW_VIDEO)
    all_total_stats.append(ov_quant_total)
    all_frame_stats.extend(ov_quant_frame)

    # 保存结果
    if Config.SAVE_CSV:
        save_stats_to_csv(all_frame_stats, Config.CSV_SAVE_PATH)

    # 打印对比结果
    # print_final_comparison(all_total_stats)

    # 释放资源
    video_cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ 所有测试完成，资源已释放")


if __name__ == "__main__":
    main()

