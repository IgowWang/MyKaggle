__author__ = 'Igor'
import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum  # 基于牛顿方程式的随机梯度下降
from nolearn.lasagne import NeuralNet

F_TRAIN = 'data/training.csv'
F_TEST = 'data/test.csv'


def load(test=False, cols=None):
    '''
    如果test为真则读取test数据，否则读取训练数据
    '''
    fname = F_TEST if test else F_TRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # Image数据需要进行切分
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=" "))

    if cols:  # 获取子列
        df = df[list[cols] + ["Image"]]

    print(df.count())  # 打印每一列的数量
    df = df.dropna()  # 删除有缺失值的行

    X = np.vstack(df['Image'].values)
    # normalize
    X = X.astype(np.float32)
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

    if not test:  # 只有训练集才有目标列
        y = df[df.columns[:-1]].values
        y = y.astype(np.float32)
        y = MinMaxScaler(feature_range=(-1, 1)).fit_transform(y)  # scale taget in [-1,1]
        # 图像的像素大小是96*96
        X, y = shuffle(X, y, random_state=42)  # 打乱训练集
    else:
        y = None

    return X, y


X, y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))

net1 = NeuralNet(
    layers=[  # 三层神经网络
              ('input', layers.InputLayer),
              ('hidden', layers.DenseLayer),
              ('output', layers.DenseLayer)],

    # layer 参数
    input_shape=(None, 9216),  # 96*96 pexels
    hidden_num_units=100,
    output_nonlinearity=None,
    output_num_units=30,  # 30 target values

    # 最优化方法
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_mementum=0.9,

    regression=True,
    max_epochs=400,
    verbose=1,
)

net1.fit(X, y)
