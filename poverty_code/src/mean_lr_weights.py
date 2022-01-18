'''
计算所得的权重系数的均值，oddsratio，以及标准误（实际上是不同实验所得权重的标准差）
'''
import os 
import json
import pandas as pd
import numpy as np
from data_gen import xindices
c_dir = "35"
l1 = 0.001 # use l1 to do variable selection
l2 = 0.
gamma = -0.5
T = 0.15 # 小于T的权重被丢弃

weight_file_path = f"output/classification_results_torch/{c_dir}/weights/weights_gamma{gamma}_l1{l1}_l2{l2}_withweight_new.txt"

var_weights = [] 
with open(weight_file_path, "r") as f:
    contents = f.readlines()
    for aline in contents:
        aline = aline.strip().split(",")
        var_weights.append([float(x) for x in aline])
var_weights = np.asarray(var_weights)
variables = [] # 变量名称
d = json.load(open("data/attrs.json", "r"))

attr_name_mapping = {}
for x in d:
    attr_name_mapping[x["attr"]] = x

for x in xindices:
    varname = ""
    if x in attr_name_mapping:
        varname = attr_name_mapping[x]["name"]
    else:
        x = x.split("_")
        varname1 = attr_name_mapping[x[0]]["name"]
        varname2 = ""
        if "mapping" in attr_name_mapping[x[0]] and x[1] in attr_name_mapping[x[0]]["mapping"]:
            varname2 = attr_name_mapping[x[0]]["mapping"][x[1]]
        else:
            varname2 = attr_name_mapping[x[0]]["default"][x[1]]
        varname = varname1.split("-")[-1] + '-' + varname2
    variables.append(varname)
variables = np.asarray(variables)
variables_ = np.empty(len(variables)+1).astype(str)
variables_[:-1] = variables
variables_[-1] = "常量"
# 计算所有模型的权重均值
mean_weights = np.mean(var_weights, axis=0)
std_error_weights = np.std(var_weights, axis=0)
odds_weights = np.exp(mean_weights)

S = np.abs(mean_weights) >= T
mean_weights = mean_weights[S]
mean_weights = np.asarray(["%.3g" % x for x in mean_weights])
std_error_weights = std_error_weights[S]
std_error_weights = np.asarray(["%.3g" % x for x in std_error_weights])
odds_weights = odds_weights[S]
odds_weights = np.asarray(["%.3g" % x for x in odds_weights])
variables_ = variables_[S]

data = np.empty((variables_.shape[0], 4)).astype(str)
data[:, 0] = variables_.astype(str)
data[:, 1] = mean_weights.astype(str)
data[:, 2] = std_error_weights.astype(str)
data[:, 3] = odds_weights.astype(str)

df = pd.DataFrame(data, columns=["变量", "Beta_i", "Std. Error", "Odds Ratio"])
df.to_csv(f"output/classification_results_torch/{c_dir}/weights_stat_{c_dir}.csv")

print(mean_weights)
print(std_error_weights)
print(odds_weights)
print(variables_)







