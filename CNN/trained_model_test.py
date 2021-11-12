import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from model import CNN
from my_dataset import MyMnistDataset

transform = transforms.Compose([
    transforms.Resize([28, 28]),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 载入自己的数据集
dataset = MyMnistDataset(root='../my_mnist_dateset', transform=transform)
test_loader = DataLoader(dataset=dataset, shuffle=False)

# 生成卷积神经网络并载入训练好的模型
model = CNN()
model.load_state_dict(torch.load("cnn_trained_model.pth"))


def test():
    correct = 0
    total = 0
    print("label       predicted")
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print("{}          {}".format(int(labels.item()), predicted.data.item()))

        print('CNN trained model： accuracy on my_mnist_dataset set:%d %%' % (100 * correct / total))


if __name__ == '__main__':
    test()
