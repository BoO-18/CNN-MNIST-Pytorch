import os
import torch
from PIL import Image
import torch.nn as nn
import numpy
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from PIL import ImageEnhance
import torch.nn.functional as F


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
            nn.ReLU(),              # 激励函数
            nn.MaxPool2d(kernel_size=2),         # 对2*2区域做最大池化操作，输出尺寸为（16，14，14）
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),     # 操作同上，输入为（16，14，14）
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


if __name__ == '__main__':
    Net_Cnn = torch.load('./models/CNN_MNIST_52.pkl')

    Test_Data = torchvision.datasets.MNIST(root='./mnist/', train=False)
    test_x = torch.unsqueeze(Test_Data.test_data, dim=1).type(torch.FloatTensor)[:-1] / 255.
    # 取测试集数据，将数据维度由(10000, 28, 28)变为(10000, 1, 28, 28)，取值范围为(0,1)
    test_y = Test_Data.test_labels[:-1]  # 取测试集标签数据

    # test_output, last_layer = Net_Cnn(test_x)
    # # print(len(test_output))
    # prediction = torch.max(test_output, 1)[1].data.numpy()
    # accuracy = float((prediction == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    # print('test accuracy: %.4f' % accuracy)

    images = Image.open('camImage-8.png')
    # 增强图像对比度
    print(images.size)
    enh_con = ImageEnhance.Contrast(images)
    contrast = 2.0
    images = enh_con.enhance(contrast)

    images = images.resize((28, 28))
    print(images.size)
    images = images.convert('L')
    transform = torchvision.transforms.ToTensor()
    images = transform(images)

    plt.imshow(images[0].numpy(), cmap='gray')
    plt.show()
    images = images.resize(1, 1, 28, 28)
    print(images.size)

    output, x = Net_Cnn(images)
    output = F.softmax(output)
    prediction = torch.max(output, 1)[1].data.numpy()
    print(output)
    print(prediction)

    # test_output, _ = Net_Cnn(test_x[:1])
    # pred_y = torch.max(test_output, 1)[1].data.numpy()
    # print(test_output)
    # print(pred_y, 'prediction number')
    # print(test_y[:1].numpy(), 'real number')


