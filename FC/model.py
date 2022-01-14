import torch


# 定义网络结构
class FC(torch.nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.l1 = torch.nn.Linear(784, 15)
        self.l2 = torch.nn.Linear(15, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)

        return x
