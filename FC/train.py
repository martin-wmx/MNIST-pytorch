import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import FC

# 查看是否有cuda如果没有，则用cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# 将输入数据标准化,ToTensor将数据转换为张量，Normalize将数据标准化，其中0.13047是均值，0.3081是方差，这两个数据是经验值
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)

test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)

# 生成全连接网络模型实例
model = FC()
model.to(device)

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


# 训练函数
def train(epoch):
    model.train()
    train_bar = tqdm(train_loader)
    for data in train_bar:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 正向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 权重更新
        optimizer.step()
        # 进度条描述训练进度
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)

        # 如果训练到第10轮，学习率就变为之前的0.1倍
        if epoch == 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1


# 验证函数
def validate(epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 得到预测值
            _, predicted = torch.max(outputs.data, dim=1)
            # 判断是否预测正确
            correct += (predicted == labels).sum().item()

            total += labels.size(0)

            # 进度条描述训练进度
            test_bar.desc = "validate epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        print('accuracy on validate set:%d %%\n' % (100 * correct / total))


if __name__ == '__main__':
    # 训练周期 cnn
    epochs = 30

    for i in range(epochs):
        train(i)

        validate(i)

    torch.save(model.state_dict(), "fc_trained_model.pth")
