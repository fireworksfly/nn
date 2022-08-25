import torch
from d2l import torch as d2l
from torch.utils import data
from torch import nn


def data_loader(data_arrays, batch_size, is_train):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == '__main__':
    true_w = torch.tensor([4.1, 2.1])
    true_b = torch.tensor(2.4)
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    data_iter = data_loader((features, labels), 20, True)

    # 定义神经网络一层
    net = nn.Sequential(nn.Linear(2, 1))
    # 初始化线性层的参数
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    # 定义损失函数
    loss = nn.MSELoss()
    # 实例化SGD
    trainer = torch.optim.SGD(net.parameters(), lr=0.02)

    num_epochs = 6

    for epoch in range(num_epochs):
        for X, y in data_iter:
            lo = loss(net(X), y)  # 反向传播只会根据之前的式子来进行更新， 将之前的式子覆盖了，所以可以直接更新。
            trainer.zero_grad()
            lo.backward()
            trainer.step()
        lo = loss(net(features), labels)
        print(f'epoch{epoch + 1}, loss {lo:f}')
        print(net[0].weight.data)
