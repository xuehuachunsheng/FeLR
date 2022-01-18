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

## Normalization for all data
X = preprocessing.scale(X, axis=0)

model = sklinear.LogisticRegression(multi_class="multinomial", class_weight="balanced", solver="saga", penalty="elasticnet", l1_ratio=1)

model.fit(X, Y)
print("Fit over")
Y_pred = model.predict(X[Y==4, :])
print(Y_pred)
