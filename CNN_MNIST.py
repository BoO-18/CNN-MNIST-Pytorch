import os
import torch
import time
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm
import torch.nn.functional as F

# 超参数设置
Epoch = 10
Batch_Size = 50
LR = 0.001
Download_MNIST = False   # 监督参数，当未下载数据集是使用True，已下载时使用False

# 检查下载mnist数据集的路径是否存在
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = False

# 装载训练集
Train_Data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(), download=Download_MNIST)
# train参数为True表示读取训练集，False表示读取测试集
Train_Loader = Data.DataLoader(dataset=Train_Data, batch_size=Batch_Size, shuffle=True)     # 建立数据迭代器

Test_Data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(Test_Data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.
# 取测试集数据，将数据维度由(10000, 28, 28)变为(10000, 1, 28, 28)，取值范围为(0,1)
test_y = Test_Data.test_labels[:2000]     # 取测试集标签数据


class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Sequential(     # 输入尺寸为（1，28，28）
            nn.Conv2d(
                in_channels=1,      # 输入维度
                out_channels=16,    # 输出维度
                kernel_size=5,      # 卷积核尺寸，一般用奇数
                stride=1,           # 卷积核移动速度：每步一单位
                padding=2,          # 边填充，为了保持尺寸不变，应选择（kernel_size - 1）/ 2
            ),  # 输出尺寸(16, 28, 28)
            nn.BatchNorm2d(16),
            nn.ReLU(),              # 激励函数
            nn.MaxPool2d(kernel_size=2),         # 对2*2区域做最大池化操作，输出尺寸为（16，14，14）
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),     # 操作同上，输入为（16，14，14）
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     # 输出为（32，7，7）
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)


if __name__ == '__main__':
    Cnn = CNN1()
    Optimizer = torch.optim.Adam(Cnn.parameters(), lr=LR)   # 定义优化函数
    Loss_Func = nn.CrossEntropyLoss()                       # 定义损失函数

    t0 = time.time()
    for epoch in range(Epoch):
        for step, (input, target) in enumerate(Train_Loader):
            output = Cnn(input)[0]
            # print(len(output))
            loss = Loss_Func(output, target)
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()

            if step % 50 == 0:
                test_output, last_layer = Cnn(test_x)
                # print(len(test_output))
                prediction = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((prediction == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '|Step:', step, '| train loss: %.4f' % loss.data.numpy(),
                      '| test accuracy: %.4f' % accuracy)
                # 记录loss与accuracy
                # f = open('./data/loss_CNN_50.txt', 'a')
                # f.writelines(['Epoch: ', str(epoch), '   ',
                #               'Step: ', str(step), '   ',
                #               'train loss: %.4f' % loss.data.numpy(), '   ',
                #               'test accuracy: %.4f' % accuracy, '\n'])
                # f.close()
                if epoch == 0 and step == 0:
                    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                    plot_only = 500
                    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                    labels = test_y.numpy()[:plot_only]
                    plot_with_labels(low_dim_embs, labels)
        if epoch == 9:
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            plot_only = 500
            low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
            labels = test_y.numpy()[:plot_only]
            plot_with_labels(low_dim_embs, labels)

    t1 = time.time()
    print('Use time:', t1 - t0)
    # torch.save(Cnn, './models/CNN_MNIST_52_soft.pkl')
