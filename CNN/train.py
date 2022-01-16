import os.path

import torch
import argparse
import pandas as pd
import numpy as np
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import CNN


def parse_opt():
    parser = argparse.ArgumentParser(description='ResNet-MNIST')
    parser.add_argument('--epochs', type=int, default=30, help='input total epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='dataloader batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='optimizer weight_decay')

    args = parser.parse_args()
    return args


def train(epoch):
    total_loss = 0
    total_correct = 0
    total_data = 0
    global iteration

    model.train()
    train_bar = tqdm(train_loader)
    for data in train_bar:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()
        # 正向传播
        outputs = model(images)
        _, predicted = torch.max(outputs.data, dim=1)
        total_correct += torch.eq(predicted, labels).sum().item()
        # 计算损失
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        # 反向传播
        loss.backward()
        # 权重更新
        optimizer.step()

        total_data += labels.size(0)
        iteration = iteration + 1

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f} iteration:{}".format(epoch + 1,
                                                                               args.epochs,
                                                                               loss,
                                                                               iteration)
    # 更新学习率
    scheduler.step()

    loss = total_loss / len(train_loader)
    acc = 100 * total_correct / total_data
    train_loss.append(loss)
    train_acc.append(acc)

    print('accuracy on train set:%d %%' % acc)


# 验证函数
def validate(epoch):
    total_loss = 0
    total_correct = 0
    total_data = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # 正向传播
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total_correct += torch.eq(predicted, labels).sum().item()

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            total_data += labels.size(0)

            # 进度条描述训练进度
            test_bar.desc = "validate epoch[{}/{}]".format(epoch + 1,
                                                           args.epochs)

        loss = total_loss / len(train_loader)
        acc = 100 * total_correct / total_data
        validate_loss.append(loss)
        validate_acc.append(acc)

        print('accuracy on validate set:%d %%\n' % acc)


if __name__ == "__main__":
    args = parse_opt()
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 由于神经网络中数据对象为tensor，所以需要用transform将普通数据转换为tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 训练数据集，torchvision中封装了数据集的下载方式，调用下面函数就会自动下载
    train_dataset = datasets.MNIST(root='../dataset/', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    # 测试数据集
    test_dataset = datasets.MNIST(root='../dataset/', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    # 生成神经网络实例
    model = CNN()
    model.to(device)

    # 损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 设置动态学习率
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train_loss = []
    train_acc = []
    validate_loss = []
    validate_acc = []
    iteration = 1

    for i in range(args.epochs):
        train(i)
        validate(i)

    torch.save(model.state_dict(), "./{}/CNN.pth".format(device_type))
    epoch = np.arange(1, args.epochs + 1)
    dataframe = pd.DataFrame({'epoch': epoch,
                              'train loss': train_loss,
                              'train accuracy': train_acc,
                              'validate loss': validate_loss,
                              'validate accuracy': validate_acc
                              })
    dataframe.to_csv(r"./{}/loss&acc.csv".format(device_type))
