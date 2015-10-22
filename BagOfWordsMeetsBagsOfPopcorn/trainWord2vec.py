__author__ = 'Igor'
import pickle
import logging
from gensim.models import word2vec


# def hash32(value):
#     return hash(value) & 0xffffffff


logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

# 读取句子
with open("data/sentences.pickle", "rb") as f:
    sentences = pickle.load(f)
f.close()

# 参数的选择
num_features = 300  # 词向量的维度
min_word_count = 40  # 最小词频
num_workers = 4  # 并行的线程数量
context = 10  # 窗口大小
downsampling = 1e-3  # downsample setting for frequent words

# 初始化训练模型
print("初始化模型...")
model = word2vec.Word2Vec(sentences, workers=num_workers,
                          size=num_features, min_count=min_word_count,
                          window=context, sample=downsampling)
model.init_sims(replace=True)
model_name = "300features_40minwords_10context"
model.save(model_name)
