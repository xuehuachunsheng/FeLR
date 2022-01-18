# micro-P micro-R micro-F1 according to the data
import os
import sys

import numpy as np
def get_confusion_matrices(file_path):
    rets = None
    with open(file_path, "r") as f:
        s = f.readlines()
        rets = np.zeros((len(s), 2, 2))
        for i in range(len(s)):
            s[i] = s[i].strip().split(", ")
            matrix = np.zeros((2, 2)) # 混淆矩阵
            neg = float(s[i][0]) 
            pos = float(s[i][1])
            if neg < 0.5:
                matrix[1, 1] += 1 # TN
            else:
                matrix[1, 0] += 1 # FP
            if pos < 0.5:
                matrix[0, 1] += 1 # FN
            else:
                matrix[0, 0] += 1 # TP
            rets[i, ...] = matrix
    return rets

if __name__ == "__main__":
    ## Parameters
    class_idx = [2, 4]
    gamma = -0.5
    l1 = 0.001
    l2 = 0.
    firth = False
    # End of parameters

    c_dir = "35" if class_idx == [3, 5] else "24"
    # file_path = f"output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_l1{l1}_l2{l2}_withweight.txt"
    file_path = f"output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_l1{l1}_l2{l2}_withweight_new.txt"
    if firth:
        file_path = f"output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_firth.txt"
    # 给定一个文件，构造出所有的混淆矩阵, 构成一个列表

    # Nx2x2 tensor
    confusion_matrices = get_confusion_matrices(file_path)

    # 2x2 matrix
    micro_matrix = np.sum(confusion_matrices, axis=0)

    print(micro_matrix)
    
    print(f"TP: {micro_matrix[0, 0]}")
    print(f"FN: {micro_matrix[0, 1]}")
    print(f"FP: {micro_matrix[1, 0]}")
    print(f"TN: {micro_matrix[1, 1]}")
    microp = micro_matrix[0, 0] / (micro_matrix[0, 0] + micro_matrix[1, 0])
    print("micro-P: %.4g" % microp)
    micror = micro_matrix[0, 0] / (micro_matrix[0, 0] + micro_matrix[0, 1])
    print("micro-R: %.4g" % micror)
    microf1 = 2 * microp * micror / (microp + micror)
    print("micro-F1: %.4g" % microf1)
