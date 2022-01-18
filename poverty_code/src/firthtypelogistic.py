import os
import sys
from data_gen import get_dataset
import numpy as np
import pandas as pd
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
models = keras.models
layers = keras.layers
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
print_logit_weights = False # 是否打印logit模型中的权重参数
write_logit_weights_to_file = False # 是否将得到的权重参数写入文件，在多次试验中，就存储多次试验的值,每一列表示不同的独立试验，每一行表示不同的指标
out_weights_file_path = f"output/focallogisticregression_weights-class_idx-{str(class_idx)}_{str(weights)}_gamma-{gamma}_l1-{l1}_l2-{l2}.csv"
ite_num = 1500
### End of PARAMS   

############
# In firth type, the regularization item should add the offset of the model
############
class FirthTypeRegularization(keras.regularizers.Regularizer):
    '''
    Using Fisher information matrix to do penalization
    param: Data, n x d matrix, with n samples and d dimensions (the last dimension is vector 1)
    '''
    def __init__(self, data): 
        self.data = tf.constant(data, dtype=tf.float32)
        self.dataT = tf.constant(np.transpose(data), dtype=tf.float32)
        self.delta = tf.eye(data.shape[1], num_columns=data.shape[1]) * 1e-7

    def __call__(self, theta):
        theta = tf.squeeze(theta)
        theta = theta[:, None]
        ret = tf.matmul(self.data, theta) #  n x d ** d x 1
        __pi__ = tf.sigmoid(ret) + 1e-10 # n x 1
        __pi__ = tf.squeeze(__pi__)
        diag_matrix = tf.diag(tf.multiply(__pi__, tf.ones_like(__pi__) - __pi__)) # n x n
        multi1 = tf.matmul(self.dataT, diag_matrix)
        multi = tf.matmul(multi1, self.data)
        # This line does work
        multi = tf.add(multi, self.delta)
        det_ret = tf.matrix_determinant(multi)
        return 0.5 * tf.log(det_ret+1e-10) # scalar value

    # getconfig

def weighted_focal_loss(weights, gamma=2.):
    epsilon = 1.e-7
    gamma = float(gamma)
    def _weighted_focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pos_loss = -y_true * tf.math.pow(tf.ones_like(y_pred) - y_pred, gamma) * tf.math.log(y_pred)
        neg_loss = -(tf.ones_like(y_true) - y_true) * tf.math.pow(y_pred, gamma) * tf.math.log(tf.ones_like(y_pred) - y_pred)
        return tf.reduce_mean(weights[1] * pos_loss + weights[0] * neg_loss)
        # return tf.reduce_mean(pos_loss + neg_loss)
    return _weighted_focal_loss

# Make one-hot encoder
def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

all_logit_weights = []
all_logit_bias = []
for i in range(num_experiments):

    train_X, train_y, test_X, test_y = get_dataset(class_idx=class_idx).__next__()
    
    #### Add the augment 1 vector on train_X and test_X
    # train_X = np.concatenate([train_X, np.ones((train_X.shape[0], 1))], axis=1)
    # test_X = np.concatenate([test_X, np.ones((test_X.shape[0], 1))], axis=1)
    ####

    train_y_ohe = one_hot_encode_object_array(train_y)
    test_y_ohe = one_hot_encode_object_array(test_y) # 必须保证每个类别都有
    train_y = train_y_ohe[:, 1] # 由于是二分类的，所以只保留其中一个类别的信息就可以了
    test_y = test_y_ohe[:, 1]

    # Model construction
    model = models.Sequential()
    # Not use bias item but the train_X should be augmented with a vector 1.
    model.add(layers.Dense(1, input_shape=(train_X.shape[1],), use_bias=False, kernel_regularizer=FirthTypeRegularization(data=train_X), name="dense1"))
    # model.add(layers.Dense(1, input_shape=(train_X.shape[1],), use_bias=True, kernel_regularizer=keras.regularizers.L1L2(l1=l1,l2=l2), name="dense1"))
    model.add(layers.Activation('sigmoid'))
    # loss_ = bin_focal_loss(gamma=2., alpha=0.999) # it seems does not work
    # loss_ = weighted_ce_loss
    loss_ = weighted_focal_loss(weights=weights, gamma=gamma)
    model.compile(loss=loss_, metrics=['accuracy'], optimizer='adam')

    # Actual modelling
    model.fit(train_X, train_y, verbose=0, batch_size=train_X.shape[0], epochs=ite_num)
    score, accuracy = model.evaluate(test_X, test_y[:, None], batch_size=1, verbose=0)
    pred_y = model.predict(test_X)
    print("###############The prediction score (The final element represents the suspect household): ")
    print(np.squeeze(pred_y))
    logit_weights = np.asarray(model.get_layer("dense1").get_weights()[0])
    # logit_bias = np.asarray(model.get_layer("dense1").get_weights()[1])
    logit_weights = np.squeeze(logit_weights)
    logit_bias = logit_weights[-1]
    logit_weights = logit_weights[:-1]
    logit_weights = np.asarray([float("%.3g" % x) for x in logit_weights])
    logit_bias = np.squeeze(logit_bias)
    logit_bias = float("%.3g" % logit_bias)
    # print(model.get_layer("dense1").get_weights())
    # print(np.concatenate([pred_y, test_y], axis=1))
    print("Test fraction correct (NN-Score) = {:.2f}".format(score))
    print("Test fraction correct (NN-Accuracy) = {:.2f}".format(accuracy))
    print()
    if print_logit_weights:
        print("###############The weights and bias: ")
        print(logit_weights)
        print(logit_bias)
    all_logit_weights.append(logit_weights)
    all_logit_bias.append(logit_bias)

if write_logit_weights_to_file:
    from data_gen import d, xindices, attr_name_mapping
    assert len(xindices) == train_X.shape[1]
    column_names = []
    for x in xindices:
        if "_" in x:
            c_name, c_index = x.split("_")
            str_name = attr_name_mapping[c_name]["name"]
            str_value = attr_name_mapping[c_name]["mapping"].get(c_index, "")
            if not str_value:
                str_value = attr_name_mapping[c_name]["default"][c_index]
            column_names.append(str_name + "_" + str_value)
        else:
            column_names.append(attr_name_mapping[x]["name"])
    column_names.append("模型偏置")
    assert len(column_names) == train_X.shape[1] + 1
    nd_weights = np.transpose(np.asarray(all_logit_weights))
    nd_bias = np.asarray(all_logit_bias)[None, :]
    nd_weight_bias = np.concatenate([nd_weights, nd_bias], axis=0)
    nd_names = np.asarray(column_names)[:, None]
    combines_ret = np.concatenate([nd_names, nd_weight_bias], axis=1)
    ret_colums = ["指标"]
    ret_colums.extend([f"Experiment-{i+1}" for i in range(num_experiments)])
    df = pd.DataFrame(combines_ret, columns=ret_colums)
    df.to_csv(out_weights_file_path)

    
