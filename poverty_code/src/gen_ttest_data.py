"""
Run batch shell script:

batch 1:
python3 src/gen_ttest_data.py --class_idx 35 --gamma 0.0 & 
python3 src/gen_ttest_data.py --class_idx 35 --gamma 0.1 & 
python3 src/gen_ttest_data.py --class_idx 35 --gamma 0.2 & 
python3 src/gen_ttest_data.py --class_idx 35 --gamma 0.3 & 
python3 src/gen_ttest_data.py --class_idx 35 --gamma 0.4 & 
python3 src/gen_ttest_data.py --class_idx 35 --gamma 0.5 & 
python3 src/gen_ttest_data.py --class_idx 35 --gamma 0.6 & 
python3 src/gen_ttest_data.py --class_idx 35 --gamma 0.7 & 
python3 src/gen_ttest_data.py --class_idx 35 --gamma 0.8 & 
python3 src/gen_ttest_data.py --class_idx 35 --gamma 0.9 & 
python3 src/gen_ttest_data.py --class_idx 35 --gamma 1.0 & 
python3 src/gen_ttest_data.py --class_idx 35 --gamma 1.1 & 
python3 src/gen_ttest_data.py --class_idx 35 --gamma 1.2 & 

batch 2:
python3 src/gen_ttest_data.py --class_idx 24 --gamma 0.0 & 
python3 src/gen_ttest_data.py --class_idx 24 --gamma 0.1 & 
python3 src/gen_ttest_data.py --class_idx 24 --gamma 0.2 & 
python3 src/gen_ttest_data.py --class_idx 24 --gamma 0.3 & 
python3 src/gen_ttest_data.py --class_idx 24 --gamma 0.4 & 
python3 src/gen_ttest_data.py --class_idx 24 --gamma 0.5 & 
python3 src/gen_ttest_data.py --class_idx 24 --gamma 0.6 & 
python3 src/gen_ttest_data.py --class_idx 24 --gamma 0.7 & 
python3 src/gen_ttest_data.py --class_idx 24 --gamma 0.8 & 
python3 src/gen_ttest_data.py --class_idx 24 --gamma 0.9 & 
python3 src/gen_ttest_data.py --class_idx 24 --gamma 1.0 & 
python3 src/gen_ttest_data.py --class_idx 24 --gamma 1.1 & 
python3 src/gen_ttest_data.py --class_idx 24 --gamma 1.2 & 

"""
# This program does not use GPU because of the compatibility between the CUDA version and tensorflow version
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
from data_gen import get_dataset
import numpy as np
import pandas as pd
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
import argparse

from data_gen import all_combinations_gen

import tensorflow as tf
from tensorflow import keras
models = keras.models
layers = keras.layers
np_utils = keras.utils

### Begin of PARAMS
parser = argparse.ArgumentParser()
parser.add_argument("--class_idx", type=str, choices=["24", "35"], help="one of [\"24\", \"35\"]")
parser.add_argument("--gamma", type=float, help="the gamma value, >0 numeric")
args = parser.parse_args()
weights35 = [506 / (498 * 2), 506 / (8 * 2)]
weights24 = [425 / (420 * 2), 425 / (5 * 2)]
weights = weights24 # 类别加权，稀有事件的类别权重要大，如果weights = [1, 1]，表示不加权
class_idx = [2, 4] # [3, 5]：表示非贫困户和疑似漏评户分类，[2, 4]表示脱贫户和疑似错退户分类
if args.class_idx == "35":
    weights = weights35
    class_idx = [3, 5] 
gamma = args.gamma # 为0表示是经典logistic regression model, >0表示使用focal loss，范围建议[0, 2]

l1 = 0.01 # l1正则参数，决定所选择的特征，范围建议[0., 1.]
l2 = 0. # l2正则参数, 范围建议[0., 1.]
out_acc_file_path = f"output/gamma_changed_ttest_data/class{args.class_idx}/accs_gamma{gamma}_l1{l1}_l2{l2}.txt"
ite_num = 1500
print("Params: ")
print(f"\t weights = {weights}")
print(f"\t class_idx = {class_idx}")
print(f"\t gamma = {gamma}")
### End of PARAMS   

 
def weighted_focal_loss(weights, gamma=2.):
    epsilon = 1.e-7
    gamma = float(gamma)
    def weighted_focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pos_loss = -y_true * tf.math.pow(tf.ones_like(y_pred) - y_pred, gamma) * tf.math.log(y_pred)
        neg_loss = -(tf.ones_like(y_true) - y_true) * tf.math.pow(y_pred, gamma) * tf.math.log(tf.ones_like(y_pred) - y_pred)
        return weights[1] * pos_loss + weights[0] * neg_loss
    return weighted_focal_loss

# Make one-hot encoder
def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

i = 0
accs = []
from data_gen import test1_gen
for train_X, train_y, test_X, test_y in all_combinations_gen(class_idx):
# for train_X, train_y, test_X, test_y in test1_gen(class_idx, 100):
    print(f"gamma: {gamma}, {i}th experiment")
    i += 1
    if i > 100:
        break
    # This process do not have random factors
    train_y_ohe = one_hot_encode_object_array(train_y)
    test_y_ohe = one_hot_encode_object_array(test_y) # 必须保证每个类别都有
    train_y = train_y_ohe[:, 1] # 由于是二分类的，所以只保留其中一个类别的信息就可以了
    test_y = test_y_ohe[:, 1]
    # test_y = np.asarray([1])
    model = models.Sequential()
    model.add(layers.Dense(1, input_shape=(171,), kernel_regularizer=keras.regularizers.L1L2(l1=l1, l2=l2), name="dense1"))
    model.add(layers.Activation('sigmoid'))
    loss_ = weighted_focal_loss(weights=weights, gamma=gamma)
    model.compile(loss=loss_, metrics=['accuracy'], optimizer='adam')
    # Actual modelling 
    model.fit(train_X, train_y, verbose=0, batch_size=train_X.shape[0], epochs=ite_num, shuffle=True)
    # pred_y = model.predict(test_X)
    score, accuracy = model.evaluate(test_X, test_y[:, None], batch_size=1, verbose=0)
    accs.append(float("%.3g" % accuracy))
    # print(accs[-1], "\t", np.squeeze(pred_y))

with open(out_acc_file_path, "w") as f:
    s_accs = ",".join([str(x) for x in accs])
    f.write(s_accs)
    
