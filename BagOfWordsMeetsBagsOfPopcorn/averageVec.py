__author__ = 'igor'
import numpy as np
from loadData import *
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def makeFeatureVec(words, model, num_features):
    # 给定一个段落，平均每个单词的向量

    # 初始化一个空的arrary
    featureVec = np.zeros((num_features,), dtype="float32")

    nwords = 0

    index2word_set = set(model.index2word)

    # 遍历所有的词，如果词出现在model中则累加
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])


    # 除以向量的总数
    featureVec = np.divide(featureVec, nwords)

    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    '''
    给定一个评论的集合，返回它们的向量表示
    :param reviews:
    :param model:
    :param numfeatures:
    :return:
    '''
    counter = 0

    reviewsFeatreVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:

        if counter % 1000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))

        reviewsFeatreVecs[counter] = makeFeatureVec(review, model, num_features)

        counter += 1
    return reviewsFeatreVecs


model = word2vec.Word2Vec.load("300features_40minwords_10context")
train, clean_train_reviews = load(remove_stopwords=True)
test, clean_test_reviews = load(test=True, remove_stopwords=True)

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, 300)
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, 300)


# 训练
forest = RandomForestClassifier(n_estimators=100, verbose=True)
forest = forest.fit(trainDataVecs, train["sentiment"])

result = forest.predict(testDataVecs)

output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("avg_w2v.csv", index=False, quoting=3)
