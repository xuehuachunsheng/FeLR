import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import torch
print(f"Using torch: {torch.__version__}")
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tensorflow import keras
from data_gen import get_dataset
np_utils = keras.utils

### Begin of PARAMS
use_focalloss = True
class_idx = [3, 5] # [3, 5]：表示非贫困户和疑似漏评户分类，[2, 4]表示脱贫户和疑似错退户分类
num_experiments = 9 # 应该大于等于稀有事件出现的次数，最好是稀有事件出现次数的倍数
weights35 = [506 / (498 * 2), 506 / (8 * 2)]
weights24 = [425 / (420 * 2), 425 / (5 * 2)]
weights = weights24 if class_idx==[2, 4] else weights35# 类别加权，稀有事件的类别权重要大，如果weights = [1, 1]，表示不加权
gamma = 0 # 为0表示是经典logistic regression model, >0表示使用focal loss，范围建议[0, 2]
l1 = 0.
l2 = 0.
firth = True
print_logit_weights = False # 是否打印logit模型中的权重参数
write_logit_weights_to_file = False # 是否将得到的权重参数写入文件，在多次试验中，就存储多次试验的值,每一列表示不同的独立试验，每一行表示不同的指标
out_weights_file_path = f"output/focallogisticregression_weights-class_idx-{str(class_idx)}_{str(weights)}_gamma-{gamma}_l1-{l1}_l2-{l2}.csv"
ite_num = 1500
### End of PARAMS   

class WeightedFocalLoss(nn.Module):
    def __init__(self, weights, gamma, data, theta, l1=l1, l2=l2, firth=firth):
        super().__init__()
        self.np_data = data
        self.weights = weights
        self.gamma = float(gamma)
        self.data = torch.tensor(data).float() # Numpy type with n x d, does not contain the augmented item
        self.dataT = torch.tensor(np.transpose(data)).float()
        if torch.cuda.is_available():
            self.data = self.data.cuda()
            self.dataT = self.dataT.cuda()
        self.theta = theta["weights"] # theta is a tensor contains [coefficients, bias] of the logistic model,
        self.bias = theta["bias"]
        self.l1 = l1 # If l1 regularization
        self.l2 = l2 # If l2 regularization
        self.firth=firth # if firth type regularization
        
    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=1e-7, max=1-(1e-7))
        pos_loss = -y_true * torch.pow(1-y_pred, gamma) * torch.log(y_pred)
        neg_loss = -(1-y_true) * torch.pow(y_pred, gamma) * torch.log(1-y_pred)
        L = torch.mean(weights[1] * pos_loss + weights[0] * neg_loss) # 极大似然, sum
        
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
        super(LogisticRegression,self).__init__()
        self.lr=nn.Linear(171,1)   #相当于通过线性变换y=x*T(A)+b可以得到对应的各个系数
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

for i in range(num_experiments):
    
    train_X, train_y, test_X, test_y = get_dataset(class_idx=class_idx).__next__()
    
    train_y_ohe = one_hot_encode_object_array(train_y)
    test_y_ohe = one_hot_encode_object_array(test_y) # 必须保证每个类别都有
    train_y = train_y_ohe[:, [1]] # 由于是二分类的，所以只保留其中一个类别的信息就可以了
    test_y = test_y_ohe[:, [1]] # [1] can not be 1!
    logistic_model=LogisticRegression()
    if torch.cuda.is_available():
        logistic_model.cuda()

    #定义损失函数和优化器
    criterion = WeightedFocalLoss(weights, gamma, train_X, logistic_model.get_params())
    optimizer=torch.optim.Adam(logistic_model.parameters(), lr=1e-3)

    # Do fitting
    
    for epoch in range(ite_num):
        # Shuffle the data
        rand_seq = np.arange(train_X.shape[0])
        np.random.shuffle(rand_seq)
        train_X = train_X[rand_seq, :]
        train_y = train_y[rand_seq]
        # To tensor
        train_X_tensor = torch.from_numpy(train_X).float()
        train_y_tensor = torch.from_numpy(train_y).float()
        
        if torch.cuda.is_available():
            x_data=Variable(train_X_tensor).cuda()
            y_data=Variable(train_y_tensor).cuda()
        else:
            x_data=Variable(train_X_tensor)
            y_data=Variable(train_y_tensor)

        out=logistic_model(x_data)  #根据逻辑回归模型拟合出的y值
        
        loss=criterion(out,y_data)  #计算损失函数
        print_loss=loss.data.item()  #得出损失函数值
        mask=out.ge(0.5).float()  #以0.5为阈值进行分类
        correct=(mask==y_data).sum()  #计算正确预测的样本个数
        acc=correct.item()/x_data.size(0)  #计算精度
        # Gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # for param in logistic_model.parameters():
    #     print(param)
    with torch.no_grad():
        test_X = torch.from_numpy(test_X).float()
        if torch.cuda.is_available():
            pred_Y = logistic_model(test_X.cuda())
            print(pred_Y.cpu())
        else:
            pred_Y = logistic_model(test_X)
            print(pred_Y)
