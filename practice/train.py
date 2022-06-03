import time

import torch.optim.optimizer
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 引入模型  同一文件夹下
from model import CIFAR10_Model

###############
# 完整的训练套路 #
###############

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../CIFAR10", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../CIFAR10", train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 查看数据集的情况
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))  # 格式化字符串
print("测试数据集的长度为：", test_data_size)

# 使用Dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络模型
model = CIFAR10_Model()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2  # 0.01
optimizer = torch.optim.SGD(model.parameters(), learning_rate)

# 设置训练网络的参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter('logs')

# 训练
start_time = time.time()
for i in range(epoch):
    print("===>第 {} 轮训练开始----".format(i + 1))

    # 训练模型开始
    # model.train()  # 训练模式：不一定要写才生效, 只有特殊层需要 Dropout,BN
    for data in train_dataloader:
        imgs, targets = data
        outputs = model(imgs)
        # 计算误差
        loss = loss_fn(outputs, targets)
        # 设置优化器优化模型
        optimizer.zero_grad()  # 数据清零
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器：梯度下降
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("训练次数:{}, Time: {}".format(total_train_step, (end_time - start_time)))
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))  # item把tensor转为数字
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 训练测试: 不需要进行调优
    # model.eval()  # 测试模式
    total_test_loss = 0
    # TODO 优化：分类问题使用正确率来衡量
    total_accuracy = 0  # 整体正确的个数
    with torch.no_grad():  # 不需要梯度优化
        for data in test_dataloader:
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            line_max = outputs.argmax(1)  # argmax(1)行上找最大，argmax(0)列上找最大，输出下标
            accuracy = (line_max == targets).sum()  # 正确的个数
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的Accuracy：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # 保存每一轮训练的模型
    torch.save(model, "./model/model_{}.pth".format(epoch))
    print("模型已保存")

writer.close()
