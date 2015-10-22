__author__ = 'igor'
from sklearn.cluster import KMeans
import time
from  gensim.models import word2vec
import pickle

model = word2vec.Word2Vec.load("300features_40minwords_10context")
start = time.time()

word_vectors = model.syn0
num_clusters = int(word_vectors.shape[0] / 5)

kmeans_clustering = KMeans(n_clusters=num_clusters, verbose=True)
idx = kmeans_clustering.fit_predict(word_vectors)

end = time.time()
print("Time taken for K Means clustering:", end - start, "seconds")

# 创建一个词\索引字典,映射每个词和对应的类标
word_centroid_map = dict(zip(model.index2word, idx))

with open("data/clustring.pickle", "wb") as f:
    pickle.dump(idx, f)
f.close()

# For the first 10 clusters
for cluster in range(0, 10):
    print("\nCluster %d" % cluster)

    words = []
    for i in range(0, len(word_centroid_map.values())):
        if (list(word_centroid_map.values())[i] == cluster):
            words.append(list(word_centroid_map.keys())[i])
    print(words)
