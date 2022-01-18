from data_gen import get_rand_combination
from data_gen import get_dataset
from scipy import stats
import numpy as np
import os

# 获取数据
class_idx = [3, 5]
c_dir = "35"

weights35 = [506 / (498 * 2), 506 / (8 * 2)]
weights24 = [425 / (420 * 2), 425 / (5 * 2)]
# 类别加权，稀有事件的类别权重要大，如果weights = [1, 1]，表示不加权
weights = weights24 if class_idx == [2, 4] else weights35
gamma = -0.5  # 为0表示是经典logistic regression model, >0表示使用focal loss，范围建议[0, 2]
l1 = 0.001
l2 = 0.
r_seed = 2
# works only when guided_by_poverty
experiment_num = 6*10*4 if class_idx == [2, 4] else 9*10*4
weight_dir = "weights"
weights_path = f"output/classification_results_torch/{c_dir}/{weight_dir}/weights_gamma{gamma}_l1{l1}_l2{l2}.txt"


# 似然函数值
def full_log_likelihood(w, X, y):
    pii = 1 / (1 + np.exp(-np.dot(X, w).reshape(-1)))
    ll = weights[1] * np.power(1-pii, gamma) * y * np.log(pii) + \
        weights[0] * np.power(pii, gamma) * (1 - y) * np.log(1 - pii)
    return np.sum(ll)

# 
def null_log_likelihood(w, X, y):
    z = np.array([w if i == X.shape[1]-1 else 0.0 for i, w in enumerate(w.reshape(1, X.shape[1])[0])]).reshape(X.shape[1], 1)
    pii = 1 / (1 + np.exp(-np.dot(X, z).reshape(-1)))
    nll = weights[1] * np.power(1-pii, gamma) * y * np.log(pii) + \
        weights[0] * np.power(pii, gamma) * (1 - y) * np.log(1 - pii)
    return np.sum(nll)

# -2Log likelihood 
# http://web.pdx.edu/~newsomj/pa551/lectur21.htm#:~:text=The%20Maximum%20Likelihood%20function%20in,values%20with%20and%20without%20x.&text=Our%20sum%20of%20squares%20regression,the%20mean%20of%20y(%20).
# https://online.stat.psu.edu/stat462/node/207/
# With the Chi-square
def neg2loglikelihood_nullhypothesis(w, X, y):
    return -2 * (null_log_likelihood(w, X, y) - full_log_likelihood(w, X, y))

def neg2loglikelihood(w,X,y):
    return -2 * full_log_likelihood(w, X, y)

# Chi-square of logistic regression
def chisquare(w, X, y):
    pii = 1 / (1 + np.exp(-np.dot(X, w).reshape(-1)))
    return np.sum(np.power(y - pii, 2) / (pii * (1 - pii)))

def mcfadden_rsquare(w, X, y):
    return 1.0 - (full_log_likelihood(w, X, y) / null_log_likelihood(w, X, y))

def coxsnell_rsquare(w, X, y):
    return 1.0 - np.exp(2/X.shape[0] * (null_log_likelihood(w, X, y) - full_log_likelihood(w, X, y)))

def nagelkerke_rsquare(w, X, y):
    return (1.0 - np.exp(2/X.shape[0] * (null_log_likelihood(w, X, y) - full_log_likelihood(w, X, y)))) / (1 - np.exp(2/X.shape[0] * null_log_likelihood(w, X, y)))

def mcfadden_adjusted_rsquare(w, X, y):
    k = float(X.shape[1])
    return 1.0 - ((full_log_likelihood(w, X, y) - k) / null_log_likelihood(w, X, y))


# Load beta
betas = []
with open(weights_path, 'r') as f:
    betas_str = f.readlines()
    for beta in betas_str:
        beta = [float(x) for x in (beta.strip().split(','))]
        betas.append(beta)

betas = np.asarray(betas)

print("Mean interplot: ", np.mean(betas[..., -1]))
# Load data

ite = get_rand_combination(class_idx=class_idx, experiment_num=experiment_num, r_seed=r_seed)

neg2ll_nulls = []
neg2lls = []
mcfadden_r2s = []
cox_snell_r2s = []
nagelkerke_r2s = []
chisquares = []
count = 0
for train_X, train_y, _, _ in ite:
    c_beta = betas[count]
    train_X = np.hstack([train_X, np.ones((train_X.shape[0], 1), dtype=float)])
    train_y = np.asarray([0 if cy == class_idx[0] else 1 for cy in train_y])
    mcfadden_r2 = mcfadden_rsquare(c_beta, train_X, train_y)
    cox_snell_r2 = coxsnell_rsquare(c_beta, train_X, train_y)
    nagelkerke_r2 = nagelkerke_rsquare(c_beta, train_X, train_y)
    cchisquare = chisquare(c_beta, train_X, train_y)
    neg2ll_null = neg2loglikelihood_nullhypothesis(c_beta, train_X, train_y)
    neg2ll = neg2loglikelihood(c_beta, train_X, train_y)

    mcfadden_r2s.append(mcfadden_r2)
    cox_snell_r2s.append(cox_snell_r2)
    nagelkerke_r2s.append(nagelkerke_r2)
    chisquares.append(cchisquare)
    neg2ll_nulls.append(neg2ll_null)
    neg2lls.append(neg2ll)
    count += 1

mean_mcfadden_r2 = np.mean(mcfadden_r2s)
mean_coxsnell_r2 = np.mean(cox_snell_r2s)
mean_nagelkerke_r2 = np.mean(nagelkerke_r2s)
mean_chisquare = np.mean(chisquares)
mean_neg2_ll_null = np.mean(neg2ll_nulls)
mean_neg2_ll = np.mean(neg2lls)

print("Mean mcfadden_r2", mean_mcfadden_r2)
print("Mean coxsnell_r2", mean_coxsnell_r2)
print("Mean nagelkerke_r2", mean_nagelkerke_r2)
print("Mean Chi-square", mean_chisquare)
P_value = 1-stats.chi2.cdf(x=mean_chisquare,df=120)
print("p-value: ", P_value)

print("Mean -2Loglikelihood without null hypothesis", mean_neg2_ll)
print("Mean -2Loglikelihood with null hypothesis", mean_neg2_ll_null)