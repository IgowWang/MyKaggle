__author__ = 'igor'
from loadData import *
import pickle
import csv
import pandas as pd

with open("data/net2.pickle", 'rb') as f:
    net = pickle.load(f)
f.close()

X = pd.read_csv("data/training.csv")
ycols = X.columns[:-1]
Xtest, _ = load2d(test=True)
ypred = net.predict(Xtest)
ypred = ypred * 48 + 48
pred_df = pd.DataFrame(ypred, columns=ycols)

f_lookup_csv = open("data/IdLookupTable.csv", 'r', newline="");
lookcsv = csv.reader(f_lookup_csv)

with open("data/predict1.csv", "w", newline="") as fwrite:
    wirter = csv.writer(fwrite)
    wirter.writerow(['RowID', 'Location'])
    next(lookcsv)
    for line in lookcsv:
        wirter.writerow([int(line[0])] + [pred_df[line[2]][int(line[1]) - 1]])
fwrite.close()
f_lookup_csv.close()
