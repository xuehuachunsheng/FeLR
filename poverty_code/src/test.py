# 统计在不同村，不同农户类型的数量
import numpy as np
import pandas as pd
import json

def data2datawithnames():
    d = json.load(open("data/attrs.json", "r"))

    attr_name_mapping = {}
    for x in d:
        attr_name_mapping[x["attr"]] = x["name"]

    data_frame = pd.read_csv("data/数据3.csv",header=0)

    xindices = list(data_frame.columns.values)

    X = data_frame.values

    xnames = []
    for x in xindices:
        xnames.append(attr_name_mapping[x])

    df = pd.DataFrame(X, columns=xnames)
    df.to_csv("data/data3_withnames.csv")

def data2dataattrsnames():
    d = json.load(open("data/attrs.json", "r"))

    attr_name_mapping = {}
    for x in d:
        attr_name_mapping[x["attr"]] = x["name"]
    data_frame = pd.read_csv("data/数据3.csv",header=0)

    xindices = list(data_frame.columns.values)

    X = data_frame.values
    
    xnames = []
    for x in xindices:
        xnames.append(attr_name_mapping[x])

    X_str = []
    for id_, x in enumerate(X): # row
        c_x_str = []
        for i, v in enumerate(x): # column 
            v = int(v)
            if i == 0: 
                c_x_str.append(str(id_))
                continue
            if d[i]['type'] in ['分类变量', '因变量']:
                cc_mappings = {}
                for k in d[i]['mapping']:
                    cc_mappings[k] = d[i]['mapping'][k]
                if 'default' in d[i]:
                    for k in d[i]['default']:
                        cc_mappings[k] = d[i]['default'][k]
                print(d[i])
                c_x_str.append(cc_mappings[str(v)])
            else:
                c_x_str.append(str(v))
        X_str.append(c_x_str)
    
    X_str = np.asarray(X_str, dtype=str)

    df = pd.DataFrame(X_str, columns=xnames)
    df.to_csv("data/data3_withattrnames.csv")

if __name__ == "__main__":
    data2dataattrsnames()