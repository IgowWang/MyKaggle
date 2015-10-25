__author__ = 'igor'
import pickle
import numpy as np
from loadData import *
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

train, clean_train_views = load(remove_stopwords=True)
test, clean_test_views = load(test=True, remove_stopwords=True)

with open("data/clustring.pickle", "rb") as f:
    idx = pickle.load(f)
f.close()

with open("300features_40minwords_10context", "rb") as f:
    model = pickle.load(f)
f.close()

word_centroid_map = dict(zip(model.index2word, idx))


def create_bag_of_centroids(wordlist, word_centroid_map):
    num_centroids = max(word_centroid_map.values()) + 1
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")

    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1

    return bag_of_centroids


num_cluster = len(set(idx))
print(num_cluster)
train_centroids = np.zeros((train["review"].size, num_cluster))
test_centroids = np.zeros((test["review"].size, num_cluster))
# bag of centroids
counter = 0
for review in clean_test_views:
    train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter + 1

counter = 0
for review in clean_test_views:
    test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter + 1

forest = RandomForestClassifier(n_estimators=100, verbose=True)

forest.fit(train_centroids, train["sentiment"])
result = forest.predict(test_centroids)

output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("data/BagOfCentroids.csv", index=False, quoting=3)
print("Done!")