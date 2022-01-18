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
xindices = list(data_frame.columns.values)
# xindices.remove("class")
xindices.remove("ID")
xindices.remove("attr02") # 去除行政村属性
xindices.remove("attr03") # 去除组属性
features = data_frame.loc[:, xindices]
one_hot_data = pd.get_dummies(features)

one_hot_data.to_csv("data/data.csv")
