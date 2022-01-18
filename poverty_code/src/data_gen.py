import json
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
seed = False
r_seed = np.random.randint(0,1000000)
if seed: np.random.seed(r_seed)
from sklearn import preprocessing
import sklearn.linear_model as sklinear
import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.ensemble as ensemble
from sklearn.model_selection import cross_val_score
import sklearn.decomposition as skdr    

normalization = True
removeDummy = False
# Get the string mapping as the dataframe input to do one hot encoding
d = json.load(open("data\\attrs.json", "r", encoding="UTF-8"))
d = d[1:-1] # Remove the ID and the class index
nomials = {}
attr_name_mapping = {}
for x in d:
    if x["type"] == '分类变量':
        nomials[x["attr"]] = str
    attr_name_mapping[x["attr"]] = {}
    attr_name_mapping[x["attr"]]["name"] = x['name']
    if "mapping" in x:
        attr_name_mapping[x["attr"]]["mapping"] = x["mapping"]
    if "default" in x:
        attr_name_mapping[x["attr"]]["default"] = x["default"]

data_frame = pd.read_csv("data/数据3.csv",header=0, dtype=nomials)
xindices = list(data_frame.columns.values)
# xindices.remove("class")
xindices.remove("ID")
xindices.remove("attr02") # 不同村
xindices.remove("attr03") # 不同组
xindices.remove("attr44") # 移除是否负债，因为它包含在了attr45
features = data_frame.loc[:, xindices]
one_hot_data = pd.get_dummies(features)
xindices = list(one_hot_data.columns.values)
if removeDummy:
    xindices.remove("attr12_0")
    xindices.remove("attr14_0")
    xindices.remove("attr15_0")
    xindices.remove("attr16_0")
    xindices.remove("attr18_0")
    xindices.remove("attr25_0")
    xindices.remove("attr30_0")
    xindices.remove("attr33_0")
    xindices.remove("attr42_0")
    xindices.remove("attr43_0")
    xindices.remove("attr52_0")
    xindices.remove("attr35_0")

xindices.remove("class")
features = one_hot_data.loc[:, xindices]

## Train data
X = features.values
Y = one_hot_data["class"].values
Y = Y.astype(int)

print(f"Attributes: {X.shape[1]}")
## Normalization for all data
# X = preprocessing.scale(X, axis=0)

# 贫困户
X_1 = X[Y==1, :]; Y_1 = Y[Y==1]
print("贫困户数据总共：", X_1.shape[0])
# 脱贫户
X_2 = X[Y==2, :]; Y_2 = Y[Y==2]
print("脱贫户数据总共：", X_2.shape[0])
# 非贫困户
X_3 = X[Y==3, :]; Y_3 = Y[Y==3]
print("非贫困户数据总共：", X_3.shape[0])
# 疑似错退户
X_4 = X[Y==4, :]; Y_4 = Y[Y==4]
print("疑似错退户数据总共：", X_4.shape[0])
# 疑似漏评户
X_5 = X[Y==5, :]; Y_5 = Y[Y==5]
print("疑似漏评户数据总共：", X_5.shape[0])
def data_split_gen(_X_, _Y_, n_train, n_gen):
    assert _X_.shape[0] == _Y_.shape[0]
    if n_train == _X_.shape[0] - 1: # Leave one out
        count = 0
        while count < n_gen:
            test_id = count % _X_.shape[0]
            train_idx = list(np.arange(_X_.shape[0]))
            train_idx.remove(test_id)
            # Do not shuffle
            X_train = _X_[train_idx, :]
            Y_train = _Y_[train_idx]
            X_test = _X_[[test_id], :]
            Y_test = _Y_[[test_id]]
            yield [X_train, Y_train, X_test, Y_test]
            count += 1
    else:
        for _ in range(n_gen):
            rand_seq = np.arange(_X_.shape[0])
            # if seed:
            #     np.random.seed(r_seed)
            np.random.shuffle(rand_seq)

            X_train = _X_[rand_seq[:n_train], :]
            Y_train = _Y_[rand_seq[:n_train]]
            X_test = _X_[rand_seq[n_train:], :]
            Y_test = _Y_[rand_seq[n_train:]]
            yield [X_train, Y_train, X_test, Y_test]

## 贫困户数据生成器
x1_gen = data_split_gen(X_1, Y_1, X_1.shape[0] - 1, n_gen=10000)
## 脱贫户数据生成器
x2_gen = data_split_gen(X_2, Y_2, X_2.shape[0] - 1, n_gen=10000)
## 非贫困户数据生成器
x3_gen = data_split_gen(X_3, Y_3, X_3.shape[0] - 1, n_gen=10000)
## 疑似错退户数据生成器
x4_gen = data_split_gen(X_4, Y_4, X_4.shape[0] - 1, n_gen=10000)
## 疑似漏评户数据生成器
x5_gen = data_split_gen(X_5, Y_5, X_5.shape[0] - 1, n_gen=10000)
XY = [None, x1_gen, x2_gen, x3_gen, x4_gen, x5_gen]
# weights24 = np.asarray([416 / (411 * 2), 416 / (5 * 2)])
# weights35 = np.asarray([508 / (499 * 2), 508 / (9 * 2)])

def get_dataset(class_idx):
    for _ in range(1000000):
        rXtrain = []
        rYtrain = []
        rXtest = []
        rYtest = []
        for i in class_idx:
            [x1,y1,x2,y2] = XY[i].__next__()
            rXtrain.append(x1)
            rYtrain.append(y1)
            rXtest.append(x2)
            rYtest.append(y2)
        rXtrain = np.concatenate(rXtrain)
        rYtrain = np.concatenate(rYtrain)
        rXtest = np.concatenate(rXtest)
        rYtest = np.concatenate(rYtest)

        if normalization:
            n_test = rXtest.shape[0]
            rX = np.concatenate([rXtrain, rXtest], axis=0)
            rX = preprocessing.scale(rX, axis=0)
            rXtrain = rX[:-n_test, :]
            rXtest = rX[-n_test:, :]

        yield [rXtrain, rYtrain, rXtest, rYtest]


def all_combinations_gen(class_idx): 
    def _data_split_one(_X, _Y, ii):
        # Leave one out, the out sample is as the test, which index is ii.
        assert type(ii)==int
        assert _X.shape[0] == _Y.shape[0]
        train_idx = list(np.arange(_X.shape[0]))
        train_idx.remove(ii)
        _X_train = _X[train_idx, :]
        _X_test = _X[[ii], :]
        _Y_train = _Y[train_idx]
        _Y_test = _Y[[ii]]
        return [_X_train, _Y_train, _X_test, _Y_test]

    # 获取这些类别的所有训练测试集的组合情况，每个类别留出1个作为测试
    X_all = [None, X_1, X_2, X_3, X_4, X_5]
    Y_all = [None, Y_1, Y_2, Y_3, Y_4, Y_5]
    assert len(class_idx) == 2
    
    XX_1 = X_all[class_idx[0]]
    YY_1 = Y_all[class_idx[0]]
    XX_2 = X_all[class_idx[1]]
    YY_2 = Y_all[class_idx[1]]
    np.random.seed(0)
    rand_seq = np.arange(XX_1.shape[0] + XX_2.shape[0] - 2)
    for c1 in range(XX_1.shape[0]):
        [XX_1_train, YY_1_train, XX_1_test, YY_1_test] = _data_split_one(XX_1, YY_1, c1)
        for c2 in range(XX_2.shape[0]):
            [XX_2_train, YY_2_train, XX_2_test, YY_2_test] = _data_split_one(XX_2, YY_2, c2)
            XX_train = np.concatenate([XX_1_train, XX_2_train], axis=0)
            YY_train = np.concatenate([YY_1_train, YY_2_train], axis=0)
            # Do not shuffle here
            # np.random.shuffle(rand_seq)
            XX_train = XX_train[rand_seq, :]
            YY_train = YY_train[rand_seq]
            XX_test = np.concatenate([XX_1_test, XX_2_test], axis=0)
            YY_test = np.concatenate([YY_1_test, YY_2_test], axis=0)

            if normalization: # 不要忘记normalization,否则结果可能完全不对
                n_test = XX_test.shape[0]
                rX = np.concatenate([XX_train, XX_test], axis=0)
                rX = preprocessing.scale(rX, axis=0)
                XX_train = rX[:-n_test, :]
                XX_test = rX[-n_test:, :]
            yield [XX_train, YY_train, XX_test, YY_test]

def random_combinations_gen(class_idx, experiment_num=100): 
    def _data_split_one(_X, _Y, ii):
        # Leave one out, the out sample is as the test, which index is ii.
        assert type(ii)==int
        assert _X.shape[0] == _Y.shape[0]
        train_idx = list(np.arange(_X.shape[0]))
        train_idx.remove(ii)
        _X_train = _X[train_idx, :]
        _X_test = _X[[ii], :]
        _Y_train = _Y[train_idx]
        _Y_test = _Y[[ii]]
        return [_X_train, _Y_train, _X_test, _Y_test]

    # 获取这些类别的所有训练测试集的组合情况，每个类别留出1个作为测试
    X_all = [None, X_1, X_2, X_3, X_4, X_5]
    Y_all = [None, Y_1, Y_2, Y_3, Y_4, Y_5]
    assert len(class_idx) == 2
    
    XX_1 = X_all[class_idx[0]]
    YY_1 = Y_all[class_idx[0]]
    XX_2 = X_all[class_idx[1]]
    YY_2 = Y_all[class_idx[1]]
    # np.random.seed(0)
    # rand_seq = np.arange(XX_1.shape[0] + XX_2.shape[0] - 2)
    i = 0
    while i < experiment_num:
        # 分别随机取正负例的一个样本
        c1 = np.random.randint(0,XX_1.shape[0])
        c2 = np.random.randint(0,XX_2.shape[0])
        [XX_1_train, YY_1_train, XX_1_test, YY_1_test] = _data_split_one(XX_1, YY_1, c1)
        [XX_2_train, YY_2_train, XX_2_test, YY_2_test] = _data_split_one(XX_2, YY_2, c2)
    
        XX_train = np.concatenate([XX_1_train, XX_2_train], axis=0)
        YY_train = np.concatenate([YY_1_train, YY_2_train], axis=0)
        XX_test = np.concatenate([XX_1_test, XX_2_test], axis=0)
        YY_test = np.concatenate([YY_1_test, YY_2_test], axis=0)

        if normalization: # 不要忘记normalization,否则结果可能完全不对
            n_test = XX_test.shape[0]
            rX = np.concatenate([XX_train, XX_test], axis=0)
            rX = preprocessing.scale(rX, axis=0)
            XX_train = rX[:-n_test, :]
            XX_test = rX[-n_test:, :]
        yield [XX_train, YY_train, XX_test, YY_test]
        i += 1

# 只保留一个正例
def test1_gen(class_idx, ret_num=100): 
    def _data_split_one(_X, _Y, ii):
        # Leave one out, the out sample is as the test, which index is ii.
        assert type(ii)==int
        assert _X.shape[0] == _Y.shape[0]
        train_idx = list(np.arange(_X.shape[0]))
        train_idx.remove(ii)
        _X_train = _X[train_idx, :]
        _X_test = _X[[ii], :]
        _Y_train = _Y[train_idx]
        _Y_test = _Y[[ii]]
        return [_X_train, _Y_train, _X_test, _Y_test]

    # 获取这些类别的所有训练测试集的组合情况，每个类别留出1个作为测试
    X_all = [None, X_1, X_2, X_3, X_4, X_5]
    Y_all = [None, Y_1, Y_2, Y_3, Y_4, Y_5]
    assert len(class_idx) == 2
    
    XX_1 = X_all[class_idx[0]]
    YY_1 = Y_all[class_idx[0]]
    XX_2 = X_all[class_idx[1]]
    YY_2 = Y_all[class_idx[1]]
    for c2 in range(XX_2.shape[0]):
        [XX_2_train, YY_2_train, XX_2_test, YY_2_test] = _data_split_one(XX_2, YY_2, c2)

        XX_train = np.concatenate([XX_1, XX_2_train], axis=0)
        YY_train = np.concatenate([YY_1, YY_2_train], axis=0)
        XX_test = XX_2_test
        YY_test = YY_2_test

        if normalization: # 不要忘记normalization,否则结果可能完全不对
            n_test = XX_test.shape[0]
            rX = np.concatenate([XX_train, XX_test], axis=0)
            rX = preprocessing.scale(rX, axis=0)
            XX_train = rX[:-n_test, :]
            XX_test = rX[-n_test:, :]
        yield [XX_train, YY_train, XX_test, YY_test]


def get_rand_combination_poverty(class_idx, experiment_num, r_seed):
    '''
    加入贫困户数据指导分类。此时不完全生成组合，从负例中随机挑选20个样本做cv，正例还是loo。
    '''
    def _data_split_one(_X, _Y, ii):
        # Leave one out, the out sample is as the test, which index is ii.
        assert type(ii)==int
        assert _X.shape[0] == _Y.shape[0]
        train_idx = list(np.arange(_X.shape[0]))
        train_idx.remove(ii)
        _X_train = _X[train_idx, :]
        _X_test = _X[[ii], :]
        _Y_train = _Y[train_idx]
        _Y_test = _Y[[ii]]
        return [_X_train, _Y_train, _X_test, _Y_test]

    # 获取这些类别的所有训练测试集的组合情况，每个类别留出1个作为测试
    X_all = [None, X_1, X_2, X_3, X_4, X_5]
    Y_all = [None, Y_1, Y_2, Y_3, Y_4, Y_5]
    assert len(class_idx) == 2
    
    XX_1 = X_all[class_idx[0]]
    YY_1 = Y_all[class_idx[0]]
    XX_2 = X_all[class_idx[1]]
    YY_2 = Y_all[class_idx[1]]
    XX_0 = X_1 # 加入贫困户作为指导
    np.random.seed(r_seed) # 引入随机种子，方便复现
    i = 0
    while i < experiment_num:
        # 分别随机取负例的一个样本
        c1 = np.random.randint(0,XX_1.shape[0])
        [XX_1_train, YY_1_train, XX_1_test, YY_1_test] = _data_split_one(XX_1, YY_1, c1)
        # 正例样本挨个取
        for c2 in range(XX_2.shape[0]):
            [XX_2_train, YY_2_train, XX_2_test, YY_2_test] = _data_split_one(XX_2, YY_2, c2)

            # 加入贫困户数据作为训练指导
            YY_0 = (np.ones_like(Y_1) * YY_2[0]).astype(np.int32)
            XX_train = np.concatenate([XX_1_train, XX_2_train, XX_0], axis=0)
            YY_train = np.concatenate([YY_1_train, YY_2_train, YY_0], axis=0)
            
            XX_test = np.concatenate([XX_1_test, XX_2_test], axis=0)
            YY_test = np.concatenate([YY_1_test, YY_2_test], axis=0)

            if normalization: # 不要忘记normalization,否则结果可能完全不对
                n_test = XX_test.shape[0]
                rX = np.concatenate([XX_train, XX_test], axis=0)
                rX = preprocessing.scale(rX, axis=0)
                XX_train = rX[:-n_test, :]
                XX_test = rX[-n_test:, :]
            yield [XX_train, YY_train, XX_test, YY_test]
            i += 1
            if i >= experiment_num: break

def get_rand_combination(class_idx, experiment_num, r_seed):
    '''
    此时不完全生成组合，从负例中随机挑选20个样本做cv（每次cv1个正例一个负例），正例还是逐样本抽样
    '''
    def _data_split_one(_X, _Y, ii):
        # Leave one out, the out sample is as the test, which index is ii.
        assert type(ii)==int
        assert _X.shape[0] == _Y.shape[0]
        train_idx = list(np.arange(_X.shape[0]))
        train_idx.remove(ii)
        _X_train = _X[train_idx, :]
        _X_test = _X[[ii], :]
        _Y_train = _Y[train_idx]
        _Y_test = _Y[[ii]]
        return [_X_train, _Y_train, _X_test, _Y_test]

    # 获取这些类别的所有训练测试集的组合情况，每个类别留出1个作为测试
    X_all = [None, X_1, X_2, X_3, X_4, X_5]
    Y_all = [None, Y_1, Y_2, Y_3, Y_4, Y_5]
    assert len(class_idx) == 2
    
    XX_1 = X_all[class_idx[0]]
    YY_1 = Y_all[class_idx[0]]
    XX_2 = X_all[class_idx[1]]
    YY_2 = Y_all[class_idx[1]]
    np.random.seed(r_seed) # 引入随机种子，方便复现
    i = 0
    while i < experiment_num:
        # 分别随机取负例的一个样本
        c1 = np.random.randint(0,XX_1.shape[0])
        [XX_1_train, YY_1_train, XX_1_test, YY_1_test] = _data_split_one(XX_1, YY_1, c1)
        # 正例样本挨个取
        for c2 in range(XX_2.shape[0]):
            [XX_2_train, YY_2_train, XX_2_test, YY_2_test] = _data_split_one(XX_2, YY_2, c2)

            XX_train = np.concatenate([XX_1_train, XX_2_train], axis=0)
            YY_train = np.concatenate([YY_1_train, YY_2_train], axis=0)            
            XX_test = np.concatenate([XX_1_test, XX_2_test], axis=0)
            YY_test = np.concatenate([YY_1_test, YY_2_test], axis=0)

            if normalization: # 不要忘记normalization,否则结果可能完全不对
                n_test = XX_test.shape[0]
                rX = np.concatenate([XX_train, XX_test], axis=0)
                rX = preprocessing.scale(rX, axis=0)
                XX_train = rX[:-n_test, :]
                XX_test = rX[-n_test:, :]
            yield [XX_train, YY_train, XX_test, YY_test]
            i += 1
            if i >= experiment_num: break

