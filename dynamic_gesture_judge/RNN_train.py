import csv
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, num_classes):
        super(RNN, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_hidden_layers)  # (单个数据在一个时刻的特征数, 隐藏层通道数, 隐藏层数)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x, None)
        out = self.fc(out[-1, :, :]) # 取最后一个时刻的output全连接映射到分类结果
        return out

class LoadData(Dataset):
    def __init__(self, path, label_idx, train_or_test="train"):
        self.label_list = []
        self.data_path = []
        self.label_idx = label_idx
        if train_or_test == "train":
            for gesture in os.listdir(path):
                data_folder = path + '/' + gesture + '/train'
                for point_data_path in os.listdir(data_folder):
                    self.data_path.append(data_folder + '/' + point_data_path)
                    self.label_list.append(gesture)
        else:
            for gesture in os.listdir(path):
                data_folder = path + '/' + gesture + '/test'
                for point_data_path in os.listdir(data_folder):
                    self.data_path.append(data_folder + '/' + point_data_path)
                    self.label_list.append(gesture)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        img_label = self.label_list[index]
        img_path = self.data_path[index]
        keypoint = []
        with open(img_path) as f:
            reader = csv.reader(f)
            for row in reader:
                row = [float(i) for i in row]
                keypoint.append(row)
            ln = int(self.label_idx[img_label])
            td = torch.tensor(keypoint)
        return td, ln

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
            X = X.transpose(0, 1)
            X = X.cuda()
            y = y.cuda()
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch(net, train_iter, loss, updater):
    net = net.cuda()
    net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        X = X.transpose(0,1)
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

if __name__ == "__main__":
    batch_size = 45
    seq_len = 10  # 数据在时间维度上的采样数量，例如一个句子有10个单词，其为10，一个视频数据有30帧，其为30
    num_hidden_layers = 2  # 隐藏层数
    hidden_size = 64  # 隐藏层单元数
    input_size = 42  # 单个数据在一个时刻的特征数
    num_classes = 3  # 分类个数
    label_ = {"right": 0, "small": 1, "up": 2}
    DATA_PATH = './dataset/keypoint'
    train_dataset = LoadData(DATA_PATH, label_, "train")
    test_dataset = LoadData(DATA_PATH, label_, "test")
    train_iter = DataLoader(train_dataset, batch_size, shuffle=True,
                            num_workers=2)
    test_iter = DataLoader(test_dataset, batch_size, shuffle=False,
                           num_workers=2)

    net = RNN(input_size, hidden_size, num_hidden_layers, len(label_))

    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    train(net, train_iter, test_iter, loss, 100, trainer)
