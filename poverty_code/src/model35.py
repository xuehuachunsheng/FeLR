import json
import pandas as pd
import numpy as np

seed = False
r_seed = 0
if seed:
    np.random.seed(r_seed)
from sklearn import preprocessing
import sklearn.linear_model as sklinear
from sklearn.model_selection import cross_val_score

# Get the string mapping as the dataframe input to do one hot encoding
d = json.load(open("data/attrs.json", "r"))
d = d[1:-1] # Remove the ID and the class index
nomials = {}
for x in d:
    if x["type"] == '分类变量':
        nomials[x["attr"]] = str

data_frame = pd.read_csv("data/数据3.csv",header=0, dtype=nomials)
one_hot_data = pd.get_dummies(data_frame)
xindices = list(one_hot_data.columns.values)
xindices.remove("class")
xindices.remove("ID")
features = one_hot_data.loc[:, xindices]

## Train data
X = features.values
Y = one_hot_data["class"].values
Y = Y.astype(int)
## Normalization

X = preprocessing.scale(X, axis=0)

# 贫困户
X_1 = X[Y==1, :]; Y_1 = Y[Y==1]
# 脱贫户
X_2 = X[Y==2, :]; Y_2 = Y[Y==2]
# 非贫困户
X_3 = X[Y==3, :]; Y_3 = Y[Y==3]
# 疑似错退户
X_4 = X[Y==4, :]; Y_4 = Y[Y==4]
# 疑似漏评户
X_5 = X[Y==5, :]; Y_5 = Y[Y==5]

# 获取一个类别的训练测试分割
def get_rand_split(_X_, _Y_, n_train):
    rand_seq = np.arange(_X_.shape[0])
    if seed:
        np.random.seed(r_seed)
    np.random.shuffle(rand_seq)

    X_train = _X_[rand_seq[:n_train], :]
    Y_train = _Y_[rand_seq[:n_train]]
    X_test = _X_[rand_seq[n_train:], :]
    Y_test = _Y_[rand_seq[n_train:]]
    return [X_train, Y_train, X_test, Y_test]

def data_split_gen(_X_, _Y_, n_train, n_gen):
    assert _X_.shape[0] == _Y_.shape[0]
    if n_train == _X_.shape[0] - 1: # Leave one out
        for test_id in range(_X_.shape[0]):
            train_idx = list(np.arange(_X_.shape[0]))
            train_idx.remove(test_id)
            # Do not shuffle
            X_train = _X_[train_idx, :]
            Y_train = _Y_[train_idx]
            X_test = _X_[[test_id], :]
            Y_test = _Y_[[test_id]]
            yield [X_train, Y_train, X_test, Y_test]
    else:
        for _ in range(n_gen):
            rand_seq = np.arange(_X_.shape[0])
            if seed:
                np.random.seed(r_seed)
                np.random.shuffle(rand_seq)

            X_train = _X_[rand_seq[:n_train], :]
            Y_train = _Y_[rand_seq[:n_train]]
            X_test = _X_[rand_seq[n_train:], :]
            Y_test = _Y_[rand_seq[n_train:]]
            yield [X_train, Y_train, X_test, Y_test]

#################### 疑似漏评户识别
print("疑似漏评户识别实验")
n_error_back = 9
x3_gen = data_split_gen(X_3, Y_3, X_3.shape[0] - 10, n_gen=10)
x5_gen = data_split_gen(X_5, Y_5, X_5.shape[0] - 1, n_gen=0)
for i in range(n_error_back):
    [X_3_train, Y_3_train, X_3_test, Y_3_test] = x3_gen.__next__()
    [X_5_train, Y_5_train, X_5_test, Y_5_test] = x5_gen.__next__()

    X_35_train = np.concatenate([X_3_train, X_5_train], axis=0)
    Y_35_train = np.concatenate([Y_3_train, Y_5_train], axis=0)

    X_35_test = np.concatenate([X_3_test, X_5_test], axis=0)
    Y_35_test = np.concatenate([Y_3_test, Y_5_test], axis=0)

    ### Using logistic lasso model with class weighted

    logreg = sklinear.LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, class_weight="balanced")

    logreg.fit(X_35_train, Y_35_train)

    Y_35_predict = logreg.predict(X_35_test)

    for pred, real in zip(Y_35_predict, Y_35_test):
        print(f"Predict: {pred}, GroundTruth: {real}")
