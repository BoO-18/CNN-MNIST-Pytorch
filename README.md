# CNN-MNIST-Pytorch

a simple mnist classification with pytorch using Convolutional NeuralNetwork and NeuralNetwork.

### Dependencies

* python 3.6 +

* pytorch 1.0 +

* logging

* torchvision

* matplotlib


### How to use

- Train MNIST models:
  - For CNN model(you should choose a model in the code): `python CNN_MNIST.py` 
  - For NN model: `python NN_MNIST.py`

- Test our models:
  - To draw a diagram for our models: `python plot_linechart.py`
  - To test our models on realistic image: `python USE_CNN.py`

### Some results


**Performance**

Results of some comparative tests.

|Model |Accuracy  |Loss  |
|----|-----| -----|
| NN(NeuralNetwork) | 95.33% | 0.1687 |
| CNN with 5*5 kernel | 98.75% | 0.0065 |
| CNN with 3*3 kernel + padding | 98.20% | 0.0038 |
| CNN with 5*5 kernel + padding | 98.90% | 0.0167 |
| CNN with 5*5 kernel + BatchNorm2d | 99.15% | 0.0001 |
| CNN with 5*5 kernel + dropout | 98.50% | 0.0302 |

![add image](https://github.com/BoO-18/CNN-MNIST-Pytorch/raw/master/image/Diagram.png)


**Test on realistic image**

![add image](https://github.com/BoO-18/CNN-MNIST-Pytorch/raw/master/image/4.png)