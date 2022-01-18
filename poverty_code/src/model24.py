import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

seed = False
r_seed = 0
if seed:
    np.random.seed(r_seed)
from sklearn import preprocessing
import sklearn.linear_model as sklinear
import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.ensemble as ensemble
from sklearn.model_selection import cross_val_score
import sklearn.decomposition as skdr    

model_name = "lr"
normalize = True
combined = False
visualization = False
vis_method = "PCA"

def get_model():
    if model_name.lower() == 'lr':
        return sklinear.LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=1, class_weight="balanced")
    elif model_name.lower() == 'linearsvm':
        return svm.LinearSVC(C=1.0, penalty="l2", class_weight="balanced", dual=True)
    elif model_name.lower() == "svc":
        return svm.SVC(C=1.0, kernel="rbf", class_weight="balanced", gamma=0.2)
    elif model_name.lower() == 'decisiontree':
        return tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=2, min_samples_leaf=1, class_weight="balanced")
    elif model_name.lower() == 'randomforest':
        return ensemble.RandomForestClassifier(n_estimators=10, class_weight="balanced", min_samples_split=2, min_samples_leaf=1, criterion="gini")

# Get the string mapping as the dataframe input to do one hot encoding
d = json.load(open("data/attrs.json", "r"))
d = d[1:-1] # Remove the ID and the class index
nomials = {}
for x in d:
    if x["type"] == '分类变量' and len(x["mapping"]) > 2 and "default" not in x:
        nomials[x["attr"]] = str
    
# 二值变量不进行one hot

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

## Normalization for all data
# X = preprocessing.scale(X, axis=0)

# 贫困户
X_1 = X[Y==1, :]; Y_1 = Y[Y==1]
# 脱贫户
X_2 = X[Y==2, :]; Y_2 = Y[Y==2]
# 疑似错退户
X_4 = X[Y==4, :]; Y_4 = Y[Y==4]

if combined:
    X_4 = np.concatenate([X_1, X_4], axis=0)
    Y_4 = np.concatenate([Y_1, Y_4], axis=0)
    Y_4[Y_4 == 1] = 4

if normalize:
    X_24 = np.concatenate([X_2, X_4], axis=0)
    Y_24 = np.concatenate([Y_2, Y_4], axis=0)
    X_24 = preprocessing.scale(X_24, axis=0)
    X_2 = X_24[Y_24==2, :]
    Y_2 = Y_24[Y_24==2]
    X_4 = X_24[Y_24==4, :]
    Y_4 = Y_24[Y_24==4]

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

#################### 疑似错退户识别
n_error_back = 6
x2_gen = data_split_gen(X_2, Y_2, X_2.shape[0] - 10, n_gen=10)
x4_gen = data_split_gen(X_4, Y_4, X_4.shape[0] - 1, n_gen=10)
print("疑似错退户识别")

for i in range(n_error_back):
    print(f"Exp{i}")
    [X_2_train, Y_2_train, X_2_test, Y_2_test] = x2_gen.__next__()
    [X_4_train, Y_4_train, X_4_test, Y_4_test] = x4_gen.__next__()

    X_24_train = np.concatenate([X_2_train, X_4_train], axis=0)
    Y_24_train = np.concatenate([Y_2_train, Y_4_train], axis=0)

    X_24_test = np.concatenate([X_2_test, X_4_test], axis=0)
    Y_24_test = np.concatenate([Y_2_test, Y_4_test], axis=0)

    ### Using logistic lasso model with class weighted

    # model = get_model()
    model = sklinear.LogisticRegression(penalty="l1", solver="saga", class_weight="balanced", max_iter=1000)
    model.fit(X_24_train, Y_24_train)

    # params = model.coef_

    Y_24_predict = model.predict(X_24_test)

    for pred, real in zip(Y_24_predict, Y_24_test):
        print(f"Predict: {pred}, GroundTruth: {real}")

    # ## 对权重为非0的特征进行降维然后可视化
    # if visualization:
        
    #     weights = model.coef_[0]
    #     X_feature_selected = X_24_train[:, np.abs(weights) < 0.02]
        
    #     model = skdr.PCA(n_components=3)
    #     X_2d = model.fit_transform(X_feature_selected)

    #     plt.scatter(X_2d[Y_24_train==2,0], X_2d[Y_24_train==2,1], alpha=0.8, label="2")
    #     plt.scatter(X_2d[Y_24_train==4,0], X_2d[Y_24_train==4,1], alpha=0.8, label="4")
        
    #     plt.legend()
    #     plt.show()

    