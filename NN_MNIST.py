import os
import torch
import time
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm

# 超参数设置
Epoch = 20
Batch_Size = 50
LR = 0.01
Download_MNIST = False   # 监督参数，当未下载数据集是使用True，已下载时使用False

# 检查下载mnist数据集的路径是否存在
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = False

# 装载训练集
Train_Data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(), download=Download_MNIST)
# train参数为True表示读取训练集，False表示读取测试集
Train_Loader = Data.DataLoader(dataset=Train_Data, batch_size=Batch_Size, shuffle=True)     # 建立数据迭代器

Test_Data = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor())
Test_Loader = Data.DataLoader(Test_Data, batch_size=Batch_Size, shuffle=False)


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.N1 = nn.Sequential(
            nn.Linear(28 * 28, 512),
            # nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(512, 256),
            # nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = self.N1(x)
        output = self.out(x)
        return output


if __name__ == '__main__':
    net = NN()
    Optimizer = torch.optim.SGD(net.parameters(), lr=LR)  # 定义优化函数
    Loss_Func = nn.CrossEntropyLoss()  # 定义损失函数

    t0 = time.time()
    for epoch in range(Epoch):
        for step, (input, target) in enumerate(Train_Loader):
            input = input.view(input.size(0), -1)
            input = torch.autograd.Variable(input)
            output = net(input)
            loss = Loss_Func(output, target)
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()

            if step % 50 == 0:
                correct = 0
                for data in Test_Loader:
                    im, lable = data
                    im = im.view(im.size(0), -1)
                    im = torch.autograd.Variable(im)
                    lable = torch.autograd.Variable(lable)
                    test_output = net(im)
                    prediction = torch.max(test_output, 1)[1].data.numpy()
                    correct += (prediction == lable.data.numpy()).sum().item()
                # accuracy = float((prediction == lable.data.numpy()).astype(int).sum()) / len(Test_Data)
                accuracy = correct / len(Test_Data)
                print('Epoch: ', epoch, '|Step:', step, '| train loss: %.4f' % loss.data.numpy(),
                      '| test accuracy: %.4f' % accuracy)
                # 记录loss与accuracy
                f = open('./data/loss_NN_drop.txt', 'a')
                f.writelines(['Epoch: ', str(epoch), '   ',
                              'Step: ', str(step), '   ',
                              'train loss: %.4f' % loss.data.numpy(), '   ',
                              'test accuracy: %.4f' % accuracy, '\n'])
                f.close()

    t1 = time.time()
    print('Use time:', t1 - t0)
    # torch.save(net, './models/NN_MNIST.pkl')
