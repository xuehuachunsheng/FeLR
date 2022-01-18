import numpy as np
from data_gen import X_2, X_3, X_4, X_5, xindices
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rcParams['boxplot.flierprops.marker'] = 'x'
plt.rc('font',family='Times New Roman') 
idx_names=[
    "attr08", # Quantity of workforce
    "attr27", # Government subsidy funds for housing construction
    "attr36", # Number of critically ill paints
    "attr38", # Self raised funds for the treatment of critically illness 
    "attr35_1", # Be able to afford medical expenses for chronic diseases
    "attr52_1", # Household income per capita exceeds 3300 CNY
    ]

idx_strs = [
    "Quantity of workforce",
    "Government subsidy funds for housing construction",
    "Number of critically ill paints",
    "Self raised funds for the treatment of critically illness",
    "Be able to afford medical expenses for chronic diseases",
    "Household income per capita exceeds 3300 CNY"
]

xindices = list(xindices)
idx = [xindices.index(x) for x in idx_names]
X24 = np.vstack([X_2[..., idx], X_4[..., idx]])

rX24 = preprocessing.scale(X24, axis=0)
rX2 = rX24[:-6, ...]
rX4 = rX24[-6:, ...]

with plt.style.context(['science']):
    plt.boxplot([rX24[..., i] for i in range(len(idx_names))], labels=idx_strs, vert=False)
    for i in range(len(idx_names)):
        plt.scatter(x=rX4[..., i], y=[i+1]*6,c="r", s=100)
    #plt.xlim((-5, 5))

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 23,
    }
    plt.xlabel(r'$z$-score', font1)
    plt.show()
