__author__ = 'Igor'
from sklearn.feature_extraction.text import CountVectorizer
from  loadData import *
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

# 初始化CountVectorizer对象

# data, clean_train_review = load()
vectorizer = CountVectorizer(max_features=5000)
# # 将文本用词袋模型表示
# train_data_features = vectorizer.fit_transform(clean_train_review)
# train_data_features = train_data_features.toarray()
# print(train_data_features.shape)

# 训练随机森林模型
# forest = RandomForestClassifier(n_estimators=100)
# forest = forest.fit(train_data_features, data["sentiment"])
# print("训练完成")
# with open("data/forest.pickle", "wb") as f:
#     pickle.dump(forest, f, -1)
# f.close()
with open("data/forest.pickle", 'rb') as f1:
    forest = pickle.load(f1)
f1.close()

# 测试
test, test_clean_text = load(test=True)
test_data_features = vectorizer.fit_transform(test_clean_text)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

output.to_csv("data/forestPredict.csv", index=False, quoting=3)
