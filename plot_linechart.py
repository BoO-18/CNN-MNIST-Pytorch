import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    f = open('./data/loss_CNN_52.txt', 'r')
    x = []
    y1 = []
    i = 0
    for line in f:
        total = line.split()
        # x = total[1].split(':')
        if i % 5 == 0 and i != 0:
            tempx = (int(total[1]) * 1200 + int(total[3])) / 50
            x.append(int(tempx))
            y1.append(float(total[9]))
        if i == 239:
            tempx = (int(total[1]) * 1200 + int(total[3])) / 50
            x.append(int(tempx))
            y1.append(float(total[9]))
        i += 1
    f.close()

    f = open('./data/loss_CNN_31.txt', 'r')
    y2 = []
    i = 0
    for line in f:
        total = line.split()
        if i % 5 == 0 and i != 0:
            y2.append(float(total[9]))
        if i == 239:
            y2.append(float(total[9]))
        i += 1
    f.close()

    f = open('./data/loss_CNN_52_batch.txt')
    y3 = []
    i = 0
    for line in f:
        total = line.split()
        if i % 5 == 0 and i != 0:
            y3.append(float(total[9]))
        if i == 239:
            y3.append(float(total[9]))
        i += 1
    f.close()

    f = open('./data/loss_CNN_52_drop.txt', 'r')
    y4 = []
    i = 0
    for line in f:
        total = line.split()
        if i % 5 == 0 and i != 0:
            y4.append(float(total[9]))
        if i == 239:
            y4.append(float(total[9]))
        i += 1
    f.close()

    f = open('./data/loss_CNN_50.txt', 'r')
    y5 = []
    i = 0
    for line in f:
        total = line.split()
        if i % 5 == 0 and i != 0:
            y5.append(float(total[9]))
        if i == 239:
            y5.append(float(total[9]))
        i += 1
    f.close()

    f = open('./data/loss_NN.txt', 'r')
    y6 = []
    i = 0
    for line in f:
        total = line.split()
        if i % 10 == 0 and i != 0:
            y6.append(float(total[9]))
        if i == 239:
            y6.append(float(total[9]))
        i += 1
    f.close()

    plt.plot(x, y6, c='y', label='NN')
    plt.plot(x, y5, c='k', label='no padding')
    plt.plot(x, y1, c='r', label='5*5')
    plt.plot(x, y2, c='b', label='3*3')
    plt.plot(x, y3, c='g', label='batch')
    plt.plot(x, y4, c='m', label='drop')
    plt.legend(loc='lower right')
    plt.show()
