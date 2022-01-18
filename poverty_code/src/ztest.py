import os
from scipy.stats import ttest_rel
import statsmodels.stats.weightstats as sw
import pandas as pd
import numpy as np

base_dir = "output/gamma_changed_ttest_data/class24/all_tf"
files = os.listdir(base_dir)
files = sorted(files)
data = []
for fi in files:
    with open(os.path.join(base_dir, fi)) as f:
        dd = f.readline().split(",")
        data.append([float(x) for x in dd])

ori_model_ret = data[0]
for i in range(1, len(data)):
    
    c_model_ret = data[i]
    count1=0
    count2=0
    count3=0

    for x1, x2 in zip(ori_model_ret, c_model_ret):
        if x2 > x1:
            count1 += 1
        elif x2 < x1:
            count2 += 1
        else:
            count3 += 1
    print(f"gamma = {i/10}, {count1}, {count2}, {count3}")
    ztest, pval = sw.ztest(c_model_ret, ori_model_ret, value=0)
    
    print(ztest)
    print(pval)
    print()
