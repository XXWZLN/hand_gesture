import torch
from torch import nn
from torch.utils.data import DataLoader
from loadData import LoadData
import torchvision
from torchvision import transforms
from loadData import LoadData


class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# trans = [transforms.ToTensor()]
# trans.insert(0, transforms.Resize(28))
# trans = transforms.Compose(trans)
# mnist_train = torchvision.datasets.FashionMNIST(
#     root="./data", train=True, transform=trans)
# mnist_test = torchvision.datasets.FashionMNIST(
#     root="./data", train=False, transform=trans)
#
# train_iter = DataLoader(mnist_train, batch_size, shuffle=True,
#                             num_workers=2)
# test_iter = DataLoader(mnist_test, batch_size, shuffle=False,
#                             num_workers=2)

# model
# net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
#
# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, std=0.01)
#
# net.apply(init_weights)

class ModuleMy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Flatten(),
                                    nn.Linear(784, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 10))

        self.layer2 = nn.Sequential(nn.Linear(42, 1024),
                                    nn.ReLU(),nn.Linear(1024, 3))
                                    # nn.Linear(1024, 512),
                                    # nn.ReLU(),
                                    # nn.Linear(512, 256),
                                    # nn.ReLU(),
                                    # nn.Linear(256, 128),
                                    # nn.ReLU(),
                                    # nn.Linear(128, 3))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, inputs):
        # x = self.layer1(inputs)
        x = self.layer2(inputs)
        return x


# 测试
def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度"""
    net = net.cuda()
    net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            X = X.cuda()
            y = y.cuda()
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# 训练
def train_epoch(net, train_iter, loss, updater):
    net = net.cuda()
    net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        X = X.cuda()
        y = y.cuda()
        y_hat = net(X)
        l = loss(y_hat, y.long())
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print("epoch:{}    train_loss:{:.3f}    train_acc:{:.3f}   test_acc:{:.3f}".format(epoch, train_loss, train_acc,
                                                                                           test_acc))
    torch.save({'model': net.state_dict()}, "./model.pth")


if __name__ == '__main__':
    batch_size = 256
    label_ = {"like": 0, "ok": 1, "call": 2}
    path = "C:/Users/antarctic polar bear/Desktop/two_hand"
    train_dataset = LoadData(path, "train")
    test_dataset = LoadData(path, "test")
    train_iter = DataLoader(train_dataset, batch_size, shuffle=True,
                            num_workers=2)
    test_iter = DataLoader(test_dataset, batch_size, shuffle=False,
                           num_workers=2)

    # trans = [transforms.ToTensor()]
    # trans.insert(0, transforms.Resize(28))
    # trans = transforms.Compose(trans)
    # mnist_train = torchvision.datasets.FashionMNIST(
    #     root="./data", train=True, transform=trans)
    # mnist_test = torchvision.datasets.FashionMNIST(
    #     root="./data", train=False, transform=trans)
    #
    # train_iter = DataLoader(mnist_train, batch_size, shuffle=True,
    #                             num_workers=2)
    # test_iter = DataLoader(mnist_test, batch_size, shuffle=False,
    #                             num_workers=2)

    net = ModuleMy()
    # train
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    train(net, train_iter, test_iter, loss, 50, trainer)
