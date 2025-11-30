import torch
import torch.nn as nn
from yolo import YoloV3

def test_complete_yolov3():
    model = YoloV3(num_classes=20)

    x = torch.randn(1, 3, 416, 416)

    out = model(x)

    print("完整的YOLOV3模型输出：")
    for i, output in enumerate(out):
        print(f"尺度{i+1}输出shape:{output.shape}")
        print(f"    -批次大小：{output.shape[0]}")
        print(f"    -预测框数量：{output.shape[1]}")
        print(f"    -每个预测的参数：{output.shape[2]}")

    # ---------------------------------------- #
    # 期望输出：
    # 尺度1 shape: [1, 507, 25]  13x13x3 = 507
    # 尺度2 shape: [1, 2028, 25] 26x26x3 = 2028
    # 尺度3 shape: [1, 8112, 25] 52x52x3 = 8112
    # ---------------------------------------- #

if __name__ == "__main__":
    test_complete_yolov3()
    