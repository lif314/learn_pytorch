import torch
from torch import nn
from torch.nn import Sequential


# 定义CIFAR10分类模型
class CIFAR10_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, input):
        x = self.sequential(input)
        return x


if __name__ == '__main__':
    # 测试网络正确性: 给定输入尺寸，查看输出尺寸是否正确
    model = CIFAR10_Model()
    input = torch.ones((64, 3, 32, 32,))  # batch_size, channels, h, w
    output = model(input)
    print(output.shape)
