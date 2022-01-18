
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('science')
from sklearn.metrics import precision_recall_curve # There are some problems

gamma = -0.5
l1 = 0.001
l2 = 0.
c_dir = '35'
n_samples = 9 # 9 samples in PUPH
acc_file_path = f"output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_l1{l1}_l2{l2}_withweight.txt"

accs = []
with open(acc_file_path, "r") as f:
    contents = f.readlines()
    for x in contents:
        accs.append(float(x.strip().split(", ")[1]))
accs = np.asarray(accs)
poss = np.asarray([0, 1, 2, 5, 6, 8]).astype(int)
negs = np.asarray([3, 4, 7]).astype(int)
accs_pos = accs.reshape((-1, 9))[:, poss].reshape(-1)
X_pos = np.asarray(list(poss+1)*40)
accs_neg = accs.reshape((-1, 9))[:, negs].reshape(-1)
X_neg = np.asarray(list(negs+1)*40)
with plt.style.context(['science', 'grid']):
    plt.scatter(X_pos, accs_pos, s=6)
    plt.scatter(X_neg, accs_neg, s=6, label=f"Misclassified")
    plt.xlabel(r"Different samples")
    plt.ylabel(r"$\pi_i$")
    plt.show()
