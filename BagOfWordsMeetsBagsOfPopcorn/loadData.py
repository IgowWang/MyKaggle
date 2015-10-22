__author__ = 'Igor'
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup

TRAIN_FILE_PATH = "data/labeledTrainData.tsv"
TEST_FILE_PATH = "data/testData.tsv"


def load(test=False, remove_stopwords=False):
    if test:
        path = TEST_FILE_PATH
    else:
        path = TRAIN_FILE_PATH
    data = pd.read_csv(path, header=0, delimiter="\t", quoting=3)
    num_reviews = data["review"].size
    clean_train_reviews = []
    for i in range(num_reviews):
        if ((i + 1) % 1000 == 0):
            print("Review %d of %d" % (i + 1, num_reviews))
        clean_train_reviews.append(review_to_words(data["review"][i], remove_stopwords))
    return data, clean_train_reviews


def review_to_words(raw_review, remove_stopwords=False):
    '''
    将影评转换为词
    :param raw_review:
    :return:
    '''
    # 去除HTML标记
    review_text = BeautifulSoup(raw_review,"lxml").get_text()
    # 去除非文字信息
    letters_only = re.sub(r"[^a-zA-Z]", " ", review_text)
    # 转换成小写且按空格分隔
    words = letters_only.lower().split()
    # 在Python中查找集合的速度比查找列表的速度更快
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        # 去除停用词
        words = [w for w in words if not w in stops]
    # 用空格连接单词，返回一个字符串
    return words
