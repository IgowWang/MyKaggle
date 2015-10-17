__author__ = 'igor'
import pickle
import matplotlib.pyplot as plt
import numpy as np
from loadData import *

with open("data/net1.pickle", 'rb') as f1:
    net1 = pickle.load(f1)
f1.close()

with open("data/net2.pickle", 'rb') as f2:
    net2 = pickle.load(f2)
f2.close()

# net1保存了训练中的结果
train_loss1 = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss1 = np.array([i["valid_loss"] for i in net1.train_history_])
train_loss2 = np.array([i["train_loss"] for i in net2.train_history_])[:400]
valid_loss2 = np.array([i["valid_loss"] for i in net2.train_history_])[:400]
plt.plot(train_loss1, linewidth=3, label="train1")
plt.plot(valid_loss1, linewidth=3, label="valid1")
plt.plot(train_loss2, linewidth=3, label="train2", linestyle="--")
plt.plot(valid_loss2, linewidth=3, label="valid2", linestyle="--")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
#plt.ylim(1e-3, 1e-2)
plt.yscale("log")
plt.show()
