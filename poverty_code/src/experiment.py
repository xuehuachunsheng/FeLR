'''
Batch process shell:


batch 1:
python3 src/experiment.py --class_idx 24 --gamma -0.5 & 
python3 src/experiment.py --class_idx 35 --gamma -0.5 & 
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import torch
torch.set_num_threads(1)
print(f"Using torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Number of threads: 1")

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from tensorflow import keras
from data_gen import get_rand_combination_poverty, get_rand_combination
np_utils = keras.utils

parser = argparse.ArgumentParser()
parser.add_argument("--class_idx", type=str, choices=["24", "35"], help="one of [\"24\", \"35\"]")
parser.add_argument("--gamma", type=float, help="the gamma value")
args = parser.parse_args()

### Begin of PARAMS
ite_num = 1500
n_attrs = 133 # 去除了一些任务无关变量
r_seed = 2
guided_by_poverty = False # 是否加入贫困户数据指导训练
class_idx = [2, 4] if args.class_idx == "24" else [3, 5]# [3, 5]：表示非贫困户和疑似漏评户分类，[2, 4]表示脱贫户和疑似错退户分类
experiment_num = 6*10*4 if class_idx == [2, 4] else 9*10*4 # works only when guided_by_poverty
weights35 = [506 / (498 * 2), 506 / (8 * 2)]
weights24 = [425 / (420 * 2), 425 / (5 * 2)]
weights = weights24 if class_idx==[2, 4] else weights35 # 类别加权，稀有事件的类别权重要大，如果weights = [1, 1]，表示不加权
if guided_by_poverty:
    weights351 = [(506 + 74) / (498 * 2), (506 + 74) / ((8 + 74) * 2)]
    weights241 = [(425 + 74) / (420 * 2), (425 + 74) / ((5 + 74) * 2)]
    weights = weights241 if class_idx==[2, 4] else weights351 # 类别加权，稀有事件的类别权重要大，如果weights = [1, 1]，表示不加权

gamma = float(args.gamma) # 为0表示是经典logistic regression model, >0表示使用focal loss，<0就增加了奇怪的知识
l1 = 0.001
l2 = 0.
firth = False
if firth: 
    weights = [1, 1]

# 是否将得到的权重参数写入文件，存储多次试验的值,每一行表示不同的独立试验，每一列表示不同的指标
write_logit_weights_to_file = True 
write_accs_to_file = True
c_dir = "".join([str(x) for x in class_idx])

acc_dir = "accs_guided_by_poverty" if guided_by_poverty else "accs"
weight_dir = "weights_guided_by_poverty" if guided_by_poverty else "weights"

acc_file_path = f"output/classification_results_torch/{c_dir}/{acc_dir}/accs_gamma{gamma}_l1{l1}_l2{l2}_withweight.txt"
weight_file_path = f"output/classification_results_torch/{c_dir}/{weight_dir}/weights_gamma{gamma}_l1{l1}_l2{l2}_withweight.txt"

if weights == [1, 1]:
    acc_file_path = f"output/classification_results_torch/{c_dir}/{acc_dir}/accs_gamma{gamma}_l1{l1}_l2{l2}_withoutweight.txt"
    weight_file_path = f"output/classification_results_torch/{c_dir}/{weight_dir}/weights_gamma{gamma}_l1{l1}_l2{l2}_withoutweight.txt"

if firth:
    # l1 l2 does not work in firth type logistic regression
    acc_file_path = f"output/classification_results_torch/{c_dir}/{acc_dir}/accs_gamma{gamma}_firth.txt"
    weight_file_path = f"output/classification_results_torch/{c_dir}/{weight_dir}/weights_gamma{gamma}_firth.txt"
    
skip = 0 # 如果实验没有做完，那么需要skip已经实验过的部分

if not os.path.exists(os.path.dirname(acc_file_path)):
    os.mkdir(os.path.dirname(acc_file_path))
if not os.path.exists(os.path.dirname(weight_file_path)):
    os.mkdir(os.path.dirname(weight_file_path))
if not os.path.exists(weight_file_path):
    open(weight_file_path, "w").close()
if not os.path.exists(acc_file_path):
    open(acc_file_path, "w").close()

with open(weight_file_path, "r") as f:
    skip = len(f.readlines())
    with open(acc_file_path, "r") as f1:
        assert skip == len(f1.readlines())

f_weights = None
f_acc = None
if skip == 0:
    f_weights = open(weight_file_path, "w")
    f_acc = open(acc_file_path, "w")
else:
    f_weights = open(weight_file_path, "a")
    f_acc = open(acc_file_path, "a")
### End of PARAMS   

class WeightedFocalLoss(nn.Module):
    def __init__(self, weights, gamma, data, theta, l1=l1, l2=l2, firth=firth):
        super().__init__()
        self.np_data = data
        self.weights = weights
        self.gamma = float(gamma)
        self.data = torch.tensor(data).float() # Numpy type with n x d, does not contain the augmented item
        self.dataT = torch.tensor(np.transpose(data)).float()

        self.theta = theta["weights"] # theta is a tensor contains [coefficients, bias] of the logistic model,
        self.bias = theta["bias"]
        self.l1 = l1 # If l1 regularization
        self.l2 = l2 # If l2 regularization
        self.firth=firth # if firth type regularization
        
    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=1e-7, max=1-(1e-7))
        pos_loss = -y_true * torch.pow(1-y_pred, gamma) * torch.log(y_pred)
        neg_loss = -(1-y_true) * torch.pow(y_pred, gamma) * torch.log(1-y_pred)
        L = torch.mean(self.weights[1] * pos_loss + self.weights[0] * neg_loss) # 负极大似然的均值, sum
        
        # Add regularization item
        regularization = 0
        if self.l1: # Only works on theta but not bias
            regularization += self.l1 * torch.sum(torch.abs(self.theta))
        if self.l2: # Only works on theta but not bias
            regularization += self.l2 * torch.sum(torch.square(self.theta))
        if self.firth: # Works on both the theta and the bias, but the data does not contain augmented item(vector 1)
            L = torch.mean(pos_loss + neg_loss)
            multi = torch.matmul(self.data, torch.transpose(self.theta, 1, 0)) + self.bias
            __pi__  = torch.sigmoid(multi)
            diag = torch.diag(torch.flatten(torch.mul(__pi__, 1 - __pi__)))
            multi = torch.matmul(torch.matmul(self.dataT, diag), self.data)
            # Add a smooth variable to avoid singular
            # It does not work in tensorflow
            fisher_info = torch.det(multi + torch.from_numpy(np.eye(self.np_data.shape[1])*1e-7))
            regularization = 0.5 * torch.log(fisher_info + 1e-10)
        return L + regularization

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr=nn.Linear(n_attrs,1)   #相当于通过线性变换y=x*T(A)+b可以得到对应的各个系数
        self.sm=nn.Sigmoid()   #相当于通过激活函数的变换

    def forward(self, x):
        x=self.lr(x)
        x=self.sm(x)
        return x

    def get_params(self):
        return {"bias": self.lr.bias, "weights": self.lr.weight}

# Make one-hot encoder
def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

ite = get_rand_combination(class_idx=class_idx, experiment_num=experiment_num, r_seed=r_seed)
if guided_by_poverty:
    ite = get_rand_combination_poverty(class_idx=class_idx, experiment_num=experiment_num, r_seed=r_seed)

count = 0
for train_X, train_y, test_X, test_y in ite:
    count+=1
    if count <= skip:
        continue
    print(f"\rNumber of experiments: {count}", end="") 
    train_y_ohe = one_hot_encode_object_array(train_y)
    test_y_ohe = one_hot_encode_object_array(test_y) # 必须保证每个类别都有
    train_y = train_y_ohe[:, [1]] # 由于是二分类的，所以只保留其中一个类别的信息就可以了
    test_y = test_y_ohe[:, [1]] # [1] can not be 1!
    logistic_model=LogisticRegression()

    #定义损失函数和优化器
    criterion = WeightedFocalLoss(weights, gamma, train_X, logistic_model.get_params())
    optimizer=torch.optim.Adam(logistic_model.parameters(), lr=1e-3)
    rand_seq = [i for i in range(train_X.shape[0])]

    for epoch in range(ite_num):
        # shuffle the data with system random
        random.shuffle(rand_seq)
        train_X = train_X[rand_seq, :]
        train_y = train_y[rand_seq]
        # To tensor
        train_X_tensor = torch.from_numpy(train_X).float()
        train_y_tensor = torch.from_numpy(train_y).float()
        
        x_data=Variable(train_X_tensor)
        y_data=Variable(train_y_tensor)

        out=logistic_model(x_data)  #根据逻辑回归模型拟合出的y值
        loss=criterion(out,y_data)  #计算损失函数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_X = torch.from_numpy(test_X).float()
        pred_Y = logistic_model(test_X)
        if write_accs_to_file: # write acc to file
            np_accs = pred_Y.transpose(1,0)[0].numpy()
            r1 = "%.4g" % np_accs[0]
            r2 = "%.4g" % np_accs[1]
            f_acc.write(f"{r1}, {r2}\n")
            f_acc.flush()
        if write_logit_weights_to_file: # write weights to file
            lrweights = None
            lrbias = None
            for param in logistic_model.named_parameters():
                if param[0] == "lr.weight":
                    lrweights = param[1].data
                elif param[0] == "lr.bias":
                    lrbias = param[1].data

            lrweights = lrweights.numpy()[0]
            lrbias = lrbias.numpy()[0]
            lrweights = ["%.4g" % x for x in lrweights]
            f_weights.write(",".join(lrweights))
            f_weights.write(f",{lrbias}\n")
            f_weights.flush()
        
f_acc.close()
f_weights.close()
