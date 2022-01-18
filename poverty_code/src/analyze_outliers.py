import os
import pandas as pd
import numpy as np
import json
from data_gen import one_hot_data
from sklearn import preprocessing
# 从中取最后9个样本进行分析

# 分析哪些属性
analysis_attrs = [
"attr06", # 社会人口学特征-60岁以上老人数
"attr10", # 社会人口学特征-常年在外务工人数
"attr11", # 社会人口学特征-家庭常住人口数
"attr36", # 三保障-医疗-大病人数
"attr38", # 三保障-医疗-自付多少元用于治疗大病
"attr04_3", # 文化程度-职校、中专
"attr04_5", # 文化程度-本科（大专）及以上
"attr13_1", # 吃营养食物的时间间隔-逢年过节
"attr20_0", # 目前居住的房屋结构-没有住房
"attr20_2", # 目前居住的房屋结构-土坯
"attr20_5", # 目前居住的房屋结构-砖混
"attr23_1", # 人畜用房是否分离-是
"attr24_1", # 目前居住的住房是谁的-自己
"attr24_3", # 目前居住的住房是谁的-亲戚朋友
"attr31_4", # 享受哪些医疗保障-4种保险
"attr32_0", # 残疾类型-无残疾
"attr32_1", # 残疾类型-一级
"attr32_2", # 残疾类型-二级
"attr32_4", # 残疾类型-四级
"attr34_10", # 慢性病类型-肿瘤癌症
"attr34_5", # 慢性病类型-肠胃疾病
"attr34_7", # 慢性病类型-风湿疾病
"attr34_9", # 慢性病类型-神经系统疾病
"attr37_6", # 大病类型-肾部疾病
"attr41_3", # 饮水主要来源-泉水
"attr41_4", # 饮水主要来源-河水
"attr41_5", # 饮水主要来源-水柜
"attr46_2", # 借款来源-金融机构贷款
"attr52_1", # 人均收入是否超过3300元-是
]


d = json.load(open("data/attrs.json", "r"))
d = d[1:-1] # Remove the ID and the class index
nomials = {}
for x in d:
    if x["type"] == '分类变量':
        nomials[x["attr"]] = str

data_frame = pd.read_csv("data/puph.csv",header=0, dtype=nomials)
xindices = list(data_frame.columns.values)
xindices.remove("ID")
xindices.remove("attr02") # 不同村
xindices.remove("attr03") # 不同组
xindices.remove("attr44") # 移除是否负债，因为它包含在了attr45
features = data_frame.loc[:, xindices]
one_hot_data = pd.get_dummies(features)
xindices = list(one_hot_data.columns.values)
xindices.remove("class")

features = one_hot_data.loc[:, analysis_attrs]
X = features.values
X = preprocessing.scale(X, axis=0)
# The first line is the feature names
X_ = np.zeros((X.shape[0] + 1, X.shape[1])).astype(str)

##### 获取样本值
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X_[i + 1, j] = "%.3g" % X[i, j]

##### 获取变量名称
variables = [] # 变量名称
attr_name_mapping = {}
for x in d:
    attr_name_mapping[x["attr"]] = x
for x in analysis_attrs:
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
X_[0, :] = variables

###### 存储
features = pd.DataFrame(data=X_, columns=analysis_attrs)
features.to_csv("data/puph_norm.csv")
