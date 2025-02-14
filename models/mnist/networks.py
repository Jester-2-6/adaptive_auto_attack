import torch.nn as nn
from LUTorch.nn import memLinear, memConv2d, memReLu
from models.mnist.cw.lis_to_lut import get_mapped_lut

DEFAULT_LUT_PATH = "outputs/lut_builder/luts/"
num_epochs = 10


class TestFCN(nn.Module):
    def __init__(self):
        super(TestFCN, self).__init__()

        LUT = get_mapped_lut()

        self.fc1 = memLinear(784, 100, LUT)
        self.relu = nn.ReLU()
        self.fc2 = memLinear(100, 10, LUT)
        self.smax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.smax(x)
        return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = memReLu()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        fc2 = self.relu(x)
        x = self.fc3(fc2)
        x = self.softmax(x)
        return x, fc2


class LeNet_no_split(nn.Module):
    def __init__(self):
        super(LeNet_no_split, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu = memReLu()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class LeNetMem(nn.Module):
    def __init__(self, lut):
        super(LeNetMem, self).__init__()
        self.conv1 = memConv2d(1, 6, 5, lut)
        self.relu = memReLu()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = memConv2d(6, 16, 5, lut)
        self.fc1 = memLinear(16 * 4 * 4, 120, lut)
        self.fc2 = memLinear(120, 84, lut)
        self.fc3 = memLinear(84, 10, lut)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(-1, 16 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        fc2 = self.relu(x)
        x = self.fc3(fc2)
        x = self.softmax(x)
        return x


class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        self.relu = memReLu()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        fc2 = self.relu(x)
        x = self.fc3(fc2)
        x = self.softmax(x)
        return x, fc2


class CifarNet_no_split(nn.Module):
    def __init__(self):
        super(CifarNet_no_split, self).__init__()
        self.relu = memReLu()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # x = self.softmax(x)
        return x


class CifarNetMem(nn.Module):
    def __init__(self, lut):
        super(CifarNetMem, self).__init__()
        self.relu = memReLu()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = memConv2d(3, 32, 3, lut, padding=1)
        self.conv2 = memConv2d(32, 32, 3, lut, padding=1)
        self.conv3 = memConv2d(32, 64, 3, lut, padding=1)
        self.conv4 = memConv2d(64, 64, 3, lut, padding=1)
        self.conv5 = memConv2d(64, 128, 3, lut, padding=1)
        self.conv6 = memConv2d(128, 128, 3, lut, padding=1)
        self.fc1 = memLinear(128 * 4 * 4, 512, lut)
        self.fc2 = memLinear(512, 128, lut)
        self.fc3 = memLinear(128, 10, lut)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class CifarNetPlusNoSplit(nn.Module):
    def __init__(self):
        super(CifarNetPlusNoSplit, self).__init__()
        self.relu = memReLu()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.fc1 = nn.Linear(512 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 100)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        # x = self.softmax(x)
        return x


class CifarNetPlus(nn.Module):
    def __init__(self):
        super(CifarNetPlus, self).__init__()
        self.relu = memReLu()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        self.fc1 = nn.Linear(512 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 100)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        fc3 = self.fc3(x)
        x = self.relu(fc3)
        x = self.fc4(x)
        # x = self.softmax(x)
        return x, fc3


class CifarNetPlusMem(nn.Module):
    def __init__(self, lut):
        super(CifarNetPlusMem, self).__init__()
        self.relu = memReLu()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = memConv2d(3, 128, 3, lut, padding=1)
        self.conv2 = memConv2d(128, 128, 3, lut, padding=1)
        self.conv3 = memConv2d(128, 256, 3, lut, padding=1)
        self.conv4 = memConv2d(256, 256, 3, lut, padding=1)
        self.conv5 = memConv2d(256, 512, 3, lut, padding=1)
        self.conv6 = memConv2d(512, 512, 3, lut, padding=1)
        self.fc1 = memLinear(512 * 4 * 4, 1000, lut)
        self.fc2 = memLinear(1000, 512, lut)
        self.fc3 = memLinear(512, 256, lut)
        self.fc4 = memLinear(256, 100, lut)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        fc3 = self.fc3(x)
        x = self.relu(fc3)
        x = self.fc4(x)
        # x = self.softmax(x)
        return x, fc3

