import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5)  # Conv2d(in_channels, out_channels, kernel_size, ...)
        self.pool = nn.MaxPool2d(2, 2)  # MaxPool2d(kernel_size, stride=None, padding=0, ...)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=5)
        self.fc = nn.Linear(4 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # 28 * 28 -> 4 * 24 * 24 -> 4 * 12 * 12
        x = self.pool(F.relu(self.conv2(x)))
        # 4 * 12 * 12 -> 8 * 8 * 8 -> 8 * 4 * 4
        x = x.view(-1, 4 * 4 * 4)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    net = Net()
    input_ = torch.randn(10, 1, 28, 28)
    # print(input_)
    out = net(input_)
    print(out)