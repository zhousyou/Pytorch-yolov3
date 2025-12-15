import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
        创建基本的卷积块，包括conv + bn + leakyrelu
        输入参数：
        * in_channels: 输入的通道数
        * out_channels: 输出的通道数
        * kernel_size: 卷积核的大小
        * stride: 步长，默认为1
        * padding: 默认为0，无填充

        输出：
        卷积之后的特征层
    """
    # 
    # 输入参数：
    # 
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)

        return x
    
class ResidualBlock(nn.Module):
    """
        创建残差结构,包含一次1x1的卷积,将通道数压缩为原来的1/2;一次3x3的卷积,将通道数恢复,特征的大小不变。
        输入参数：
        * channels:输入的通道数

        输出参数:
        * x: 经过残差结果之后的特征
    """
    def __init__(self, channels):
        super().__init__()

        # 第一次经过1x1的卷积，通道数压缩为原来的1/2，大小不变
        self.conv1 = ConvBlock(in_channels=channels, out_channels=channels//2, kernel_size=1)

        # 第二次通过3x3的卷积，通道数恢复为输入的通道数，大小同样不变
        self.conv2 = ConvBlock(in_channels=channels//2, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += identity
        return x
    
class Darknet53(nn.Module):
    """
        构建完整的darknet53主干网络
        输出：输出三个特征层：[256, 52, 52], [512, 26, 26], [1024, 13, 13]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=32, kernel_size=3,
                               padding=1)
        self.conv2 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, 
                               stride=2, padding=1)
        # 创建残差块序列
        self.residual_blocks = nn.ModuleList()
        channel_list = [64, 128, 256, 512, 1024]
        repeat_list = [1, 2, 8, 8, 4]

        # 记录需要输出的层索引
        self.output_indices = []
        current_index = 0

        for i, (channel, repeat) in enumerate(zip(channel_list, repeat_list)):
            if i > 0:
                self.residual_blocks.append(ConvBlock(in_channels=channel_list[i-1],
                                                      out_channels=channel,
                                                      kernel_size=3,
                                                      stride=2,
                                                      padding=1))
                current_index += 1
            for j in range(repeat):
                self.residual_blocks.append(ResidualBlock(channel))
                if j == repeat - 1:
                    self.output_indices.append(current_index)
                current_index += 1
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        output = []

        for i, layer in enumerate(self.residual_blocks):
            x = layer(x)
            # 输出x的shape[B, C, H ,W]
            if i in self.output_indices and x.size(2) in [52, 26, 13]:
                output.append(x)
        return output


# if __name__ == "__main__":

#     input_tensor = torch.randn(1, 3, 416, 416)
#     model = Darknet53()
#     out = model(input_tensor)
#     # print(out[0].shape, out[1].shape, out[2].shape)
#     for layer in out:
#         print(layer.shape)