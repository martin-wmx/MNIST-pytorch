import torch


# 定义网络结构
class FC(torch.nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.l1(x)
        x = self.relu(x)

        x = self.l2(x)
        x = self.relu(x)

        x = self.l3(x)
        x = self.relu(x)

        x = self.l4(x)
        x = self.relu(x)

        x = self.l5(x)

        return x
