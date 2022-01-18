
# 用于解释现象
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)

x_gammaneg = 6
x_gamma0 = 10
x_gammapos = 14

x_gammaneg_points = np.asarray([[x_gammaneg, -1], [x_gammaneg, 0.75]])
x_gamma0_points = np.asarray([[x_gamma0, -1], [x_gamma0, 0.75]])
x_gammapos_points = np.asarray([[x_gammapos, -1], [x_gammapos, 0.75]])

with plt.style.context(['science']):
    mean = [0, 0]
    cov = [[1, 0], [-20, 1]]
    x0, y0 = np.random.multivariate_normal(mean, cov, 1000).T
    plt.plot(x0, y0, 'ob', markersize=10)
    plt.text(-13, -0.25, r'Negative samples', fontsize=20)
    mean = [20, 0]
    cov = [[1, 0], [20, 1]]
    x1, y1 = np.random.multivariate_normal(mean, cov, 100).T
    x1[65] = 7
    y1[65] = -0.5
    plt.plot(x1, y1, 'sr', markersize=10)
    plt.text(22, -0.4, r'Positive samples', fontsize=20)
    plt.plot(x_gammaneg_points[:, 0], x_gammaneg_points[:, 1], linestyle='--', linewidth=2)
    plt.plot(x_gamma0_points[:, 0], x_gamma0_points[:, 1], linestyle='--', linewidth=2)
    plt.plot(x_gammapos_points[:, 0], x_gammapos_points[:, 1], linestyle='--', linewidth=2)

    plt.text(5, 0.8, r'$\gamma < 0$', fontsize=25)
    plt.text(9, 0.8, r'$\gamma = 0$', fontsize=25)
    plt.text(13, 0.8, r'$\gamma > 0$', fontsize=25)
    plt.plot([5, 16], [-1.05, -1.05], "-", color="black", linewidth=3)
    plt.annotate('',xy=(4.6, -1.05),xytext=(5., -1.05),
    arrowprops=dict(arrowstyle='-|>,head_width=0.4', facecolor='black'))
    plt.text(5.5, -1.1, r'Improve the recall of positive class', fontsize=15)
    plt.show()