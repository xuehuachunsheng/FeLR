from micro_measure import get_confusion_matrices
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('science')

c_dir = "35"

micro_Rs = []
micro_Ps = []
micro_F1 = []

gammas = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# 暂时不加firth type，因为实在慢
model_names = [r"F-Baseline", r"F-WLR", r"F-WLR-$l_1$", r"F-WLR-$l_2$", r"F-WLR-$l_{12}$", r"F-WLR-$l_{\text{Firth}}$"]

##### Generate F-Baseline data
fbaseline_micro_Rs = []
fbaseline_micro_Ps = []
fbaseline_micro_F1s = []

fwlr_micro_Rs = []
fwlr_micro_Ps = []
fwlr_micro_F1s = []

fwlrl1_micro_Rs = []
fwlrl1_micro_Ps = []
fwlrl1_micro_F1s = []

fwlrl2_micro_Rs = []
fwlrl2_micro_Ps = []
fwlrl2_micro_F1s = []

fwlrl12_micro_Rs = []
fwlrl12_micro_Ps = []
fwlrl12_micro_F1s = []

fwlrlf_micro_Rs = []
fwlrlf_micro_Ps = []
fwlrlf_micro_F1s = []

for gamma in gammas:
    # F-Baseline 
    fbaseline_file_name = f"output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_l10.0_l20.0_withoutweight.txt"
    matrices = get_confusion_matrices(fbaseline_file_name)
    micro_matrix = np.sum(matrices, axis=0)
    microp = micro_matrix[0, 0] / (micro_matrix[0, 0] + micro_matrix[1, 0])
    micror = micro_matrix[0, 0] / (micro_matrix[0, 0] + micro_matrix[0, 1])
    microf1 = 2 * microp * micror / (microp + micror)

    fbaseline_micro_Ps.append(microp)
    fbaseline_micro_Rs.append(micror)
    fbaseline_micro_F1s.append(microf1)

    # F-WLR
    fwlr_file_name = f"output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_l10.0_l20.0_withweight.txt"
    matrices = get_confusion_matrices(fwlr_file_name)
    micro_matrix = np.sum(matrices, axis=0)
    microp = micro_matrix[0, 0] / (micro_matrix[0, 0] + micro_matrix[1, 0])
    micror = micro_matrix[0, 0] / (micro_matrix[0, 0] + micro_matrix[0, 1])
    microf1 = 2 * microp * micror / (microp + micror)

    fwlr_micro_Ps.append(microp)
    fwlr_micro_Rs.append(micror)
    fwlr_micro_F1s.append(microf1)

    # F-WLR-l1
    fwlrl1_file_name = f"output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_l10.001_l20.0_withweight.txt"
    matrices = get_confusion_matrices(fwlrl1_file_name)
    micro_matrix = np.sum(matrices, axis=0)
    microp = micro_matrix[0, 0] / (micro_matrix[0, 0] + micro_matrix[1, 0])
    micror = micro_matrix[0, 0] / (micro_matrix[0, 0] + micro_matrix[0, 1])
    microf1 = 2 * microp * micror / (microp + micror)

    fwlrl1_micro_Ps.append(microp)
    fwlrl1_micro_Rs.append(micror)
    fwlrl1_micro_F1s.append(microf1)

    # F-WLR-l2
    fwlrl2_file_name = f"output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_l10.0_l20.001_withweight.txt"
    matrices = get_confusion_matrices(fwlrl2_file_name)
    micro_matrix = np.sum(matrices, axis=0)
    microp = micro_matrix[0, 0] / (micro_matrix[0, 0] + micro_matrix[1, 0])
    micror = micro_matrix[0, 0] / (micro_matrix[0, 0] + micro_matrix[0, 1])
    microf1 = 2 * microp * micror / (microp + micror)

    fwlrl2_micro_Ps.append(microp)
    fwlrl2_micro_Rs.append(micror)
    fwlrl2_micro_F1s.append(microf1)

    # F-WLR-l12
    fwlrl12_file_name = f"output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_l10.0005_l20.0005_withweight.txt"
    matrices = get_confusion_matrices(fwlrl12_file_name)
    micro_matrix = np.sum(matrices, axis=0)
    microp = micro_matrix[0, 0] / (micro_matrix[0, 0] + micro_matrix[1, 0])
    micror = micro_matrix[0, 0] / (micro_matrix[0, 0] + micro_matrix[0, 1])
    microf1 = 2 * microp * micror / (microp + micror)

    fwlrl12_micro_Ps.append(microp)
    fwlrl12_micro_Rs.append(micror)
    fwlrl12_micro_F1s.append(microf1)

    # F-WLR-lf
    fwlrlf_file_name = f"output/classification_results_torch/{c_dir}/accs/accs_gamma{gamma}_firth.txt"
    matrices = get_confusion_matrices(fwlrlf_file_name)
    micro_matrix = np.sum(matrices, axis=0)
    microp = micro_matrix[0, 0] / (micro_matrix[0, 0] + micro_matrix[1, 0])
    micror = micro_matrix[0, 0] / (micro_matrix[0, 0] + micro_matrix[0, 1])
    microf1 = 2 * microp * micror / (microp + micror)

    fwlrlf_micro_Ps.append(microp)
    fwlrlf_micro_Rs.append(micror)
    fwlrlf_micro_F1s.append(microf1)

figsize=(6,5.5)
with plt.style.context(['science', 'grid']):
    #### Recall
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(gammas, fbaseline_micro_Rs, "*-", ms=5, label=r"F-Baseline")
    #ax.scatter([gammas[5]], [fbaseline_micro_Rs[5]], marker='*', s=200)
    ax.plot(gammas, fwlr_micro_Rs, "o-", ms=5, label=r"F-WLR")
    ax.plot(gammas, fwlrl1_micro_Rs, "s-", ms=5, label=r"F-WLR-$l_1$")
    ax.plot(gammas, fwlrl2_micro_Rs, "D-", ms=4, label=r"F-WLR-$l_2$")
    ax.plot(gammas, fwlrl12_micro_Rs, "x-", ms=4, label=r"F-WLR-$l_{12}$")
    ax.plot(gammas, fwlrlf_micro_Rs, "p-", ms=4, label=r"F-LR-$l_{\text{Firth}}$")

    ax.legend(title=r'methods')
    ax.set(xlabel=r'$\gamma$')
    ax.set(ylabel=r'micro-$R$')
    # ax.autoscale(tight=True)        
    # fig.savefig('figures/fig1.pdf')        

    #### Precision
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)     
    ax.plot(gammas, fbaseline_micro_Ps, "*-", ms=5, label=r"F-Baseline")
    ax.plot(gammas, fwlr_micro_Ps, "o-", ms=5, label=r"F-WLR")
    ax.plot(gammas, fwlrl1_micro_Ps, "s-", ms=5, label=r"F-WLR-$l_1$")
    ax.plot(gammas, fwlrl2_micro_Ps, "D-", ms=4, label=r"F-WLR-$l_2$")
    ax.plot(gammas, fwlrl12_micro_Ps, "x-", ms=4, label=r"F-WLR-$l_{12}$")
    ax.plot(gammas, fwlrlf_micro_Ps, "p-", ms=4, label=r"F-LR-$l_{\text{Firth}}$")
    ax.legend(title=r'methods')        
    ax.set(xlabel=r'$\gamma$')        
    ax.set(ylabel=r'micro-$P$')        
    # ax.autoscale(tight=True)        
    # fig.savefig('figures/fig1.pdf')        

    ### F1
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)        
    ax.plot(gammas, fbaseline_micro_F1s, "*-", ms=5, label=r"F-Baseline")
    ax.plot(gammas, fwlr_micro_F1s, "o-", ms=5, label=r"F-WLR")
    ax.plot(gammas, fwlrl1_micro_F1s, "s-", ms=5, label=r"F-WLR-$l_1$")
    ax.plot(gammas, fwlrl2_micro_F1s, "D-", ms=4, label=r"F-WLR-$l_2$")
    ax.plot(gammas, fwlrl12_micro_F1s, "x-", ms=4, label=r"F-WLR-$l_{12}$")
    ax.plot(gammas, fwlrlf_micro_F1s, "p-", ms=4, label=r"F-LR-$l_{\text{Firth}}$")
    
    ax.legend(title=r'methods')        
    ax.set(xlabel=r'$\gamma$')        
    ax.set(ylabel=r'micro-$F1$')
    # ax.autoscale(tight=True)        
    # fig.savefig('figures/fig1.pdf')        

plt.show()

