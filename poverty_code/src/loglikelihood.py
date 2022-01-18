# 计算-2loglikelihood

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
from data_gen import get_rand_combination

# 获取数据
class_idx = [3, 5]
c_dir = "35"

weights35 = [506 / (498 * 2), 506 / (8 * 2)]
weights24 = [425 / (420 * 2), 425 / (5 * 2)]
weights = weights24 if class_idx==[2, 4] else weights35# 类别加权，稀有事件的类别权重要大，如果weights = [1, 1]，表示不加权
gamma = -0.5 # 为0表示是经典logistic regression model, >0表示使用focal loss，范围建议[0, 2]
l1 = 0.001
l2 = 0.
r_seed = 2
experiment_num = 6*10*4 if class_idx == [2, 4] else 9*10*4 # works only when guided_by_poverty
weight_dir = "weights"
weights_path = f"output/classification_results_torch/{c_dir}/{weight_dir}/weights_gamma{gamma}_l1{l1}_l2{l2}.txt"

def loglikelihood(sample, label, beta, bias):
    pii = 1.0 / (1.0 + np.exp(-(np.dot(sample, beta) + bias)))
    return weights[1] * (1-pii)**gamma * label * np.log(pii) + weights[0] * pii**gamma * (1-label) * np.log(1-pii) 

# Load beta
betas = []
with open(weights_path, 'r') as f:
    betas_str = f.readlines()
    for beta in betas_str:
        beta = [float(x) for x in (beta.strip().split(','))]
        betas.append(beta)

betas = np.asarray(betas)

# Load data

ite = get_rand_combination(class_idx=class_idx, experiment_num=experiment_num, r_seed=r_seed)

loglikelihoods = []
count = 0
for train_X, train_y, _, _ in ite:
    c_beta = betas[count]
    sumlog_likelihood = 0.
    for i in range(len(train_X)):
        cx = train_X[i]
        cy = train_y[i]
        # cy = 0 if cy == 2 else 1 # 如果是PRPH分类， 那么2对应的是负例0，4对应的是正例1
        cy = 0 if cy == 3 else 1 # 如果是PRPH分类， 那么2对应的是负例0，4对应的是正例1
        sumlog_likelihood += loglikelihood(cx,cy,c_beta[:-1],c_beta[-1])
    loglikelihoods.append(sumlog_likelihood)
    count += 1

mean_loglikelihood = np.mean(loglikelihoods)
print("-2Log likelihood: ", -2 * mean_loglikelihood)

    