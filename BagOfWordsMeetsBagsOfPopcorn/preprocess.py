__author__ = 'Igor'
import pandas as pd
from loadData import *
import nltk
import pickle

# Load the punkt tokenizer

tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

# 读取数据
train = pd.read_csv("data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("data/testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

print("Read %d labeled train reviews,%d labeled test reviews,"
      "and %d unlabeled reviews\n"%(train["review"].size,
                                       test["review"].size, unlabeled_train["review"].size))


def review_to_sentence(review, tokenizer, remove_stopwords=False):
    # 将文本划分成句子，句子切分成词
    raw_sentences = tokenizer.tokenize(review.strip())

    # 遍历每个句子
    sentences = []
    for sents in raw_sentences:
        if len(sents) > 0:
            sentences.append(review_to_words(sents, remove_stopwords))
    return sentences


sentences = []

print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentence(review, tokenizer)
print(sentences[0])

print("Parsing sentences from unlabeled set")

for review in unlabeled_train["review"]:
    sentences += review_to_sentence(review, tokenizer)


with open("data/sentences.pickle","wb") as f:
    pickle.dump(sentences,f)
f.close()
