"""
一旦训练好了一个模型，可以使用这个进行测试
将训练好的模型应用在实际场景中
- 训练模型映射：GPU--CPU
- 注意模型输入，Batch_size
"""
import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.nn import Sequential

image_path = "./imgs/1.png"
image = Image.open(image_path)
# 4通道的图片
image = image.convert("RGB")  # 转为三通道

# 更改图片尺寸
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

img = transform(image)
print(img.shape)


# 加载模型：第一种方式保存模型，加载时需要引入模型
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


# 加载模型  -- GPU训练的模型，映射到CPU上
model = torch.load("./model/model_1.pth", map_location=torch.device("cpu"))
# print(model)

# 测试图片
# 输入需要四维。 batch_size
img = torch.reshape(img, (-1, 3, 32,32))
# img = img.cuda()   # 使用GPU训练的模型需要放在GPU上
# model.eval()

# 测试时不需要进行优化
with torch.no_grad():
    output = model(img)
    print(output)
    print(output.argmax(1))