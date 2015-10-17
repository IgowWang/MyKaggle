__author__ = 'igor'
from lasagne import layers
from lasagne.updates import nesterov_momentum  # 基于牛顿方程式的随机梯度下降
from nolearn.lasagne import NeuralNet
import pickle
import os
from loadData import *

net2 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer)],
    input_shape=(None, 1, 96, 96),  # 输入数据的维度
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30,
    output_nonlinearity=None,  # 因为是回归问题，所以输出层的激活函数是横等函数

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=1000,
    verbose=1,
)

X, y = load2d()
net2.fit(X, y)

with open("data/net2.pickle", 'wb') as f:
    pickle.dump(net2, f, -1)
f.close()
