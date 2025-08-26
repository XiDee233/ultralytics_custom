import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    in0 = torch.rand(1, 3, 640, 640, dtype=torch.float)
    out = []

    with ncnn.Net() as net:
        net.load_param("D:/Android/pyProj/ultralytics-main/ultralytics-main/runs/detect/Archie/train13/weights/best.ncnn.param")
        net.load_model("D:/Android/pyProj/ultralytics-main/ultralytics-main/runs/detect/Archie/train13/weights/best.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.squeeze(0).numpy()).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)).unsqueeze(0))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

if __name__ == "__main__":
    result = test_inference()
    print("推理输出:", result.shape)  # 确保输出维度正确
    print("推理结果:", result)  # 查看数值
