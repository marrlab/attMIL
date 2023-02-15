import os
import numpy as np
import pickle

from sklearn.model_selection import KFold


classes = []

with open("data-features/classes.txt") as f:
    clsdata = f.readlines()
    for cls in clsdata:
        classes.append(cls.strip("\n"))



# Create Folds
files = os.listdir("data-features/all")

patients = np.unique([x.split("_")[0] + "_" + x.split("_")[1] for x in files if not x.startswith(".")])


train_folds = [[] for x in range(3)]
test_folds  = [[] for x in range(3)]

for cls in classes:
    count = np.count_nonzero([p for p in patients if p.split("_")[0] == cls])
    print (cls," - " ,count)
    pat = np.array([p for p in patients if p.split("_")[0] == cls])

    kfold = KFold(3, True, 1)
    fold = []

    for fold_id , (train, test) in enumerate(kfold.split(pat)):
        p1 = pat[train]
        p2 = pat[test]

        train_folds[fold_id].extend(p1)
        test_folds[fold_id].extend(p2)


folds = [(train_folds[i], test_folds[i]) for i in range(3)]


with open("folds.pkl", "wb") as f:
    pickle.dump(folds, f)


print("finished")
