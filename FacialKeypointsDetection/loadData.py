__author__ = 'igor'
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import os

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


def load2d(test=False, cols=None):
    X, y = load(test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

