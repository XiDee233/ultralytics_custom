import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType,
    CalibrationDataReader,
    create_calibrator,
    CalibrationMethod
)
from onnxsim import simplify
import cv2
import numpy as np
import os
import json

# --------------------------
# 配置路径
# --------------------------
ORIGINAL_ONNX_PATH = r"D:\pyProject\ultralytics_custom\runs\fish\detect\Archie\yolo112025726\train7\weights\best.onnx"
CALIBRATION_IMAGES_DIR = r"D:\pyProject\ultralytics_custom\datasets\61fishs\dataset_20250603_014305_yolo\dataset\enhance\splitNew\lianghua"
QUANTIZED_OUTPUT_PATH = r"D:\pyProject\ultralytics_custom\runs\fish\detect\Archie\yolo112025726\train7\weights\best_quantized.onnx"
TEST_IMAGE_PATH = r"D:\pyProject\ultralytics_custom\test.jpg"  # 可选


# --------------------------
# 1. 数据读取器
# --------------------------
class YOLOCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_dir, input_name, input_size=(640, 640)):
        self.image_dir = image_dir
        self.input_name = input_name
        self.input_size = input_size
        self.image_list = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png'))
        ]
        self.index = 0
        print(f"找到 {len(self.image_list)} 张校准图像")

    def get_next(self):
        if self.index >= len(self.image_list):
            return None

        img_path = self.image_list[self.index]
        img = cv2.imread(img_path)
        if img is None:
            self.index += 1
            return self.get_next()

        img = cv2.resize(img, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)

        self.index += 1
        return {self.input_name: img}


# --------------------------
# 2. 模型预处理
# --------------------------
def preprocess_model(onnx_path):
    # 加载并检查模型
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    # 简化模型
    model_simp, check = simplify(model)
    if not check:
        raise ValueError("模型简化失败")

    simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
    onnx.save(model_simp, simplified_path)
    print(f"简化模型已保存: {simplified_path}")

    # 尝试优化模型
    try:
        # 对于较新版本
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = simplified_path.replace(".onnx", "_optimized.onnx")
        _ = ort.InferenceSession(simplified_path, sess_options, providers=["CPUExecutionProvider"])
        optimized_path = sess_options.optimized_model_filepath
    except:
        # 对于旧版本，使用简化后的模型
        print("使用简化模型替代优化模型")
        optimized_path = simplified_path

    print(f"优化模型已保存: {optimized_path}")
    return optimized_path


# --------------------------
# 3. 执行量化
# --------------------------
def quantize():
    # 预处理模型
    optimized_path = preprocess_model(ORIGINAL_ONNX_PATH)

    # 获取模型输入名称
    sess = ort.InferenceSession(optimized_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"模型输入名称: {input_name}")

    # 创建校准器（使用兼容参数）
    calibrator = create_calibrator(
        model=optimized_path,
        op_types_to_calibrate=["Conv", "MatMul"],
        calibrate_method=CalibrationMethod.MinMax
    )

    # 读取校准数据
    data_reader = YOLOCalibrationDataReader(CALIBRATION_IMAGES_DIR, input_name)
    calibrator.collect_data(data_reader)
    calibration_cache = calibrator.compute_range()
    print("校准数据计算完成")

    # 保存校准表
    calibration_table_path = QUANTIZED_OUTPUT_PATH.replace(".onnx", "_calibration.json")
    with open(calibration_table_path, 'w') as f:
        json.dump({
            name: {
                'min': float(tensor.min),
                'max': float(tensor.max)
            }
            for name, tensor in calibration_cache.items()
        }, f, indent=2)
    print(f"校准表已保存: {calibration_table_path}")

    # 准备量化参数
    nodes_to_quantize = None  # 可选：指定要量化的节点
    nodes_to_exclude = None  # 可选：指定要排除的节点

    # 执行动态量化（使用兼容参数）
    quantize_dynamic(
        model_input=optimized_path,
        model_output=QUANTIZED_OUTPUT_PATH,
        op_types_to_quantize=["Conv", "MatMul"],  # 指定要量化的算子类型
        per_channel=True,  # 启用逐通道量化
        reduce_range=False,  # 不减少量化范围
        weight_type=QuantType.QInt8,  # 权重量化为QInt8
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude,
        extra_options={
            "ActivationSymmetric": False,  # 激活值非对称量化
            "WeightSymmetric": True,  # 权重对称量化
            "AddQDQPairToWeight": True  # 为权重添加QDQ对
        }
    )

    print(f"量化模型已保存: {QUANTIZED_OUTPUT_PATH}")

    # 验证量化模型
    verify_model()


# --------------------------
# 4. 验证量化模型
# --------------------------
def verify_model():
    try:
        # 加载量化模型
        sess = ort.InferenceSession(QUANTIZED_OUTPUT_PATH, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        print(f"\n量化模型加载成功！输入名称: {input_name}")

        # 测试推理
        if os.path.exists(TEST_IMAGE_PATH):
            img = cv2.imread(TEST_IMAGE_PATH)
            img = cv2.resize(img, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            img = np.expand_dims(img, 0)

            # 推理
            outputs = sess.run(None, {input_name: img})
            print(f"推理成功！输出形状: {[o.shape for o in outputs]}")
    except Exception as e:
        print(f"验证失败: {str(e)}")


# --------------------------
# 运行主函数
# --------------------------
if __name__ == "__main__":
    # 确保校准图像文件夹存在
    if not os.path.exists(CALIBRATION_IMAGES_DIR):
        os.makedirs(CALIBRATION_IMAGES_DIR)
        print(f"已创建校准图像文件夹: {CALIBRATION_IMAGES_DIR}")
        print("请放入10-20张代表性图像后重新运行")
    else:
        quantize()