__author__ = 'Igor'
import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum  # 基于牛顿方程式的随机梯度下降
from nolearn.lasagne import NeuralNet
import matplotlib.pyplot as plt
import pickle


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

    X = np.vstack(df['Image'].values) / 255.
    # normalize
    X = X.astype(np.float32)

    if not test:  # 只有训练集才有目标列
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48
        y = y.astype(np.float32)
        # y = MinMaxScaler(feature_range=(-1, 1)).fit_transform(y)  # scale taget in [-1,1]
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
    input_shape=(None, 9216),  # 96*96 pxels
    hidden_num_units=100,
    output_nonlinearity=None,
    output_num_units=30,  # 30 target values

    # 最优化方法
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,
    max_epochs=400,
    verbose=1,
)

net1.fit(X, y)
with open("data/net1.pickle", 'wb') as f:
    pickle.dump(net1, f, -1)
f.close()

# net1保存了训练中的结果
train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
plt.plot(train_loss, linewidth=3, label="train")
plt.plot(valid_loss, linewidth=3, label="valid")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(1e-3, 1e-2)
plt.yscale("log")


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    # 显示图片
    axis.imshow(img, cmap="gray")
    # 画出检测点，一个参数是x抽，第二个参数是y轴
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker="x", s=10)


# 导入训练集
Xtest, _ = load(test=True)
y_pred = net1.predict(Xtest)

# figsize图像大小，6×6,单位是英寸
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(16):
    # subplot(行，列，索引)，即创建了一个4×4 16张子图，第三个参数是索引，按行
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y_pred[i], ax)
