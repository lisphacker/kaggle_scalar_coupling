import numpy as np

def score(df, ref, test):
    types = df.type.unique()
    T = len(types)

    s = 0

    for t in types:
        sel = (df.type == t).array
        nt = sel.sum()
        abssum = np.abs(ref[sel] - test[sel]).sum()
        logavg = np.log(abssum / nt)
        s += logavg

    return s / T