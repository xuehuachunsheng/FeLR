
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('science')
from sklearn.metrics import precision_recall_curve # There are some problems
import pandas as pd

df = pd.read_csv("data/puph_norm1.csv")

names = df.columns.values
X = df.values

# 选择一些列进行分析
selected_columns = [0, 1, 2, 3, 4, 8, 11, 16, 19, 28]

names = names[selected_columns]
print(names)
eng_names = [r'Number of the olds over 60',
r'Number of migrant workers',
r'Number of permanent residents',
r'Number of critically ill patients',
 r'Self raised funds for the treatment of critically ill', 
 r'Current housing structure-None', 
 r'Houses for humans and livestock are separated', 
 r'Level of disabled-Level 1', 
 r'Type of chronic disease-Cancer',
 r'Household income per capita exceeds 3300 CNY']
X = X[:, selected_columns]

# 给所有值都加随机扰动
np.random.seed(0)
R = np.random.randn(X.shape[0], X.shape[1])
# X = X + R

X3 = X[:-9, :]
X5 = X[-9:, :]
X5_pos = X5[np.asarray([0, 1, 2, 5, 6, 8]), :]
X5_neg = X5[np.asarray([3, 4, 7]), :]

x3label = [x for x in range(len(names))] * X3.shape[0]
x5poslabel = [x for x in range(len(names))] * X5_pos.shape[0]
x5neglabel = [x for x in range(len(names))] # * X5_neg.shape[0]

with plt.style.context(['science', 'grid']):
    # plt.scatter(x3label, X3.reshape(-1), s=20)
    # plt.scatter(x5poslabel, X5_pos.reshape(-1), s=20)
    # plt.scatter(x5neglabel, X5_neg[0, :], marker="x", s=20)
    # plt.scatter(x5neglabel, X5_neg[1, :], marker="x", s=20)
    # plt.scatter(x5neglabel, X5_neg[2, :], marker="x", s=20)
    # plt.xticks(x5neglabel, eng_names)

    plt.scatter(y=x5poslabel, x=X5_pos.reshape(-1), s=20)
    plt.scatter(y=x5neglabel, x=X5_neg[2, :], marker="x", s=20, label=r"Misclassified")
    plt.scatter(y=x5neglabel, x=X5_neg[1, :], marker="x", s=20, label=r"Misclassified")
    plt.scatter(y=x5neglabel, x=X5_neg[0, :], marker="x", s=20, label=r"Misclassified")
    ax = plt.gca()
    ax.tick_params(axis = 'y', which = 'major', labelsize = 8)
    
    plt.yticks(x5neglabel, eng_names, rotation=30)
    plt.xlabel(r"$z$-score")
    plt.legend()
    # plt.xlabel()

plt.show()
