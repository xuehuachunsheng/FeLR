# PCA降维后 进行数据可视化

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

seed = False
r_seed = 0
if seed:
    np.random.seed(r_seed)
from sklearn import preprocessing
import sklearn.decomposition as skdr
import sklearn.manifold as skm

model_name = "PCA"
normalize = True
combined = False

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

X_24 = np.concatenate([X_2, X_4], axis=0)
Y_24 = np.concatenate([Y_2, Y_4], axis=0)
X_24 = preprocessing.scale(X_24, axis=0)
X_2 = X_24[Y_24==2, :]
Y_2 = Y_24[Y_24==2]
X_4 = X_24[Y_24==4, :]
Y_4 = Y_24[Y_24==4]

X_124 = np.concatenate([X_1, X_2, X_4], axis=0)
Y_124 = np.concatenate([Y_1, Y_2, Y_4], axis=0)

# X_124 = preprocessing.minmax_scale(X_124, feature_range=(0,1))
X_124 = preprocessing.scale(X_124, axis=0)
X_1 = X_124[Y_124==1, :]
Y_1 = Y_124[Y_124==1]
X_2 = X_124[Y_124==2, :]
Y_2 = Y_124[Y_124==2]
X_4 = X_124[Y_124==4, :]
Y_4 = Y_124[Y_124==4]

model = skdr.PCA(n_components=2)
model = skm.TSNE(n_components=2)
model = skm.SpectralEmbedding(n_components=2)
model = skm.MDS(n_components=2)
model = skdr.SparsePCA(n_components=2)
model = skdr.DictionaryLearning(n_components=2)
model = skdr.KernelPCA(n_components=2, kernel="rbf")
# model = skdr.LatentDirichletAllocation(n_components=2)
# model = skdr.NMF(n_components=2)
model = skdr.FactorAnalysis(n_components=2)
model = skdr.FastICA(n_components=2)
model = skdr.TruncatedSVD(n_components=2)

# X_2d = model.fit_transform(X_124)
X_24_2d = model.fit_transform(X_24)
# plt.scatter(X_2d[Y_124==2,0], X_2d[Y_124==2,1], alpha=0.8, label="2")
# plt.scatter(X_2d[Y_124==4,0], X_2d[Y_124==4,1], alpha=0.8, label="4")
# plt.scatter(X_2d[Y_124==1,0], X_2d[Y_124==1,1], alpha=0.8, label="1")

plt.scatter(X_24_2d[Y_24==2,0], X_24_2d[Y_24==2,1], alpha=0.8, label="2")
plt.scatter(X_24_2d[Y_24==4,0], X_24_2d[Y_24==4,1], alpha=0.8, label="4")

plt.legend()
plt.show()