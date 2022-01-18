
n_attrs = 52

attr_idx = [f"attr_{i}" for i in range(n_attrs)]

attr_names = []
with open("attr_names.txt", 'r') as f:
    contents = f.readlines()
    for c in contents:
        attr_names.append(c.strip())

