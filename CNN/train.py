import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model import CNN
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# 由于神经网络中数据对象为tensor，所以需要用transform将普通数据转换为tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 训练数据集，torchvision中封装了mnist数据集的下载方式，调用下面函数就会自动下载
train_dataset = datasets.MNIST(root='../dataset/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)

# 测试数据集
test_dataset = datasets.MNIST(root='../dataset/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)

# 生成神经网络实例
model = CNN()

# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


# 训练函数
def train(epoch):
    model.train()
    train_bar = tqdm(train_loader)
    for data in train_bar:
        images, labels = data

        optimizer.zero_grad()

        outputs = model(images.to(device))

        loss = criterion(outputs, labels.to(device))

        loss.backward()
        optimizer.step()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)


# 验证函数
def validate(epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
            # 进度条描述训练进度
            test_bar.desc = "test epoch[{}/{}]".format(epoch + 1,
                                                       epochs)

        print('accuracy on test set:%d %%' % (100 * correct / total))


if __name__ == '__main__':
    # 训练周期
    epochs = 30

    for i in range(epochs):
        train(i)

        validate(i)

    torch.save(model.state_dict(), "cnn_trained_model.pth")
