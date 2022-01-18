# 绘制pr曲线

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('science')
from sklearn.metrics import precision_recall_curve # There are some problems


gamma = -0.5
c_dir = '24'
n_cv = 240 if c_dir == '24' else 360

def precision_recall(gt_Y, pred_Y): # 正类的precision和recall
    confusion_matrix = np.zeros((2, 2))
    for label, pred in zip(gt_Y, pred_Y):
        label = abs(label - 1) # 0 represents pos
        pred = abs(pred - 1) # 0 represents pos
        confusion_matrix[int(label), int(pred)] += 1
    TP = confusion_matrix[0, 0]
    FP = confusion_matrix[1, 0]
    FN = confusion_matrix[0, 1]
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision, recall

def pr_curve(gt_Y, pred_Y):
    precisions = []
    recalls = []
    thresholds = []
    indices = np.argsort(pred_Y)[::-1]
    s_gt_Y = gt_Y[indices]
    s_pred_Y = pred_Y[indices]
    for _, pred in zip(s_gt_Y, s_pred_Y):
        thresholds.append(pred)
        cc_pred_Y = (s_pred_Y >= pred).astype(int)
        pre, rec = precision_recall(s_gt_Y, cc_pred_Y)
        precisions.append(pre)
        recalls.append(rec)
    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    thresholds = np.asarray(thresholds)
    return precisions, recalls, thresholds

def readaccfile(file_path):
    with open(file_path, 'r') as f:
        ret = f.readlines()
        ss = []
        for x in ret:
            xx = x.strip().split(', ')
            ss.append([float(xx[0]), float(xx[1])])
        ss = np.transpose(np.asarray(ss))
        ss = ss.flatten()
        return ss.flatten() # y_pred

# Ground-truth y
Y0 = [0 for _ in range(n_cv)]
Y1 = [1 for _ in range(n_cv)]
gt_Y = np.zeros(n_cv*2).astype(int)
gt_Y[:n_cv] = np.array(Y0).astype(int)
gt_Y[n_cv:] = np.array(Y1).astype(int)

with plt.style.context(['science', 'grid']):

    # Graph 1
    fig, ax = plt.subplots()
    ## Baseline
    baseline_file_name = f'output/classification_results_torch/{c_dir}/accs/accs_gamma0.0_l10.0_l20.0_withoutweight.txt'
    y_pred = readaccfile(baseline_file_name)
    precision, recall, thresholds = pr_curve(gt_Y, y_pred)
    ax.plot(recall,precision, label=r'Baseline')
    ## F-Baseline
    fbaseline_file_name = f'output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_l10.0_l20.0_withoutweight.txt'
    y_pred = readaccfile(fbaseline_file_name)
    precision, recall, thresholds = pr_curve(gt_Y, y_pred)

    ax.plot(recall,precision, label=r'F-Baseline')
    ax.legend(title=r'methods')
    plt.xlabel(r'micro-$R$')
    plt.ylabel(r'micro-$P$')
    # # Graph 2
    fig, ax = plt.subplots()
    ## WLR
    wlr_file_name = f'output/classification_results_torch/{c_dir}/accs/accs_gamma0.0_l10.0_l20.0_withweight.txt'
    y_pred = readaccfile(wlr_file_name)
    precision, recall, thresholds = pr_curve(gt_Y, y_pred)

    ax.plot(recall,precision, label=r'WLR')
    
    ## F-WLR
    fwlr_file_name = f'output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_l10.0_l20.0_withweight.txt'
    y_pred = readaccfile(fwlr_file_name)
    precision, recall, thresholds = pr_curve(gt_Y, y_pred)

    ax.plot(recall,precision, label=r'F-WLR')
    ax.legend(title=r'methods')
    plt.xlabel(r'micro-$R$')
    plt.ylabel(r'micro-$P$')

    fig, ax = plt.subplots()
    ## WLR-l1    
    wlrl1_file_name = f'output/classification_results_torch/{c_dir}/accs/accs_gamma0.0_l10.001_l20.0_withweight.txt'
    y_pred = readaccfile(wlrl1_file_name)
    precision, recall, thresholds = pr_curve(gt_Y, y_pred)

    ax.plot(recall,precision, label=r'WLR-$l_1$')
    ax.legend(title=r'methods')

    ## F-WLR-l1    
    fwlrl1_file_name = f'output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_l10.001_l20.0_withweight.txt'
    y_pred = readaccfile(fwlrl1_file_name)
    precision, recall, thresholds = pr_curve(gt_Y, y_pred)

    ax.plot(recall,precision, label=r'F-WLR-$l_1$')
    ax.legend(title=r'methods')
    plt.xlabel(r'micro-$R$')
    plt.ylabel(r'micro-$P$')

    fig, ax = plt.subplots()
    ## WLR-l2
    wlrl2_file_name = f'output/classification_results_torch/{c_dir}/accs/accs_gamma0.0_l10.0_l20.001_withweight.txt'
    y_pred = readaccfile(wlrl2_file_name)
    precision, recall, thresholds = pr_curve(gt_Y, y_pred)

    ax.plot(recall,precision, label=r'WLR-$l_2$')
    ax.legend(title=r'methods')

    ## F-WLR-l2
    fwlrl2_file_name = f'output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_l10.0_l20.001_withweight.txt'
    y_pred = readaccfile(fwlrl2_file_name)
    precision, recall, thresholds = pr_curve(gt_Y, y_pred)

    ax.plot(recall,precision, label=r'F-WLR-$l_2$')
    ax.legend(title=r'methods')
    plt.xlabel(r'micro-$R$')
    plt.ylabel(r'micro-$P$')

    fig, ax = plt.subplots()
    ## WLR-l12
    wlrl12_file_name = f'output/classification_results_torch/{c_dir}/accs/accs_gamma0.0_l10.0005_l20.0005_withweight.txt'
    y_pred = readaccfile(wlrl12_file_name)
    precision, recall, thresholds = pr_curve(gt_Y, y_pred)

    ax.plot(recall,precision, label=r'WLR-$l_{12}$')
    ax.legend(title=r'methods')
    
    ## F-WLR-l12
    fwlrl12_file_name = f'output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_l10.0005_l20.0005_withweight.txt'
    y_pred = readaccfile(fwlrl12_file_name)
    precision, recall, thresholds = pr_curve(gt_Y, y_pred)
    ax.plot(recall,precision, label=r'F-WLR-$l_{12}$')
    ax.legend(title=r'methods')
    plt.xlabel(r'micro-$R$')
    plt.ylabel(r'micro-$P$')

    fig, ax = plt.subplots()
    ## LR-firth
    wlrlf_file_name = f'output/classification_results_torch/{c_dir}/accs/accs_gamma0.0_firth.txt'
    y_pred = readaccfile(wlrlf_file_name)
    precision, recall, thresholds = pr_curve(gt_Y, y_pred)
    ax.plot(recall,precision, label=r'WLR-$l_{\text{Firth}}$')
    ax.legend(title=r'methods')

    ## F-LR-firth
    fwlrlf_file_name = f'output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_firth.txt'
    y_pred = readaccfile(fwlrlf_file_name)
    precision, recall, thresholds = pr_curve(gt_Y, y_pred)
    ax.plot(recall,precision, label=r'F-WLR-$l_{\text{Firth}}$')
    ax.legend(title=r'methods')
    plt.xlabel(r'micro-$R$')
    plt.ylabel(r'micro-$P$')

plt.show()