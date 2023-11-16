from __future__ import print_function, division
from builtins import range
import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency

def get_p_value(T):
    if False:
        for i in range(10):
            print('nop')
    det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
    c2 = float(det) / T[0].sum() * det / T[1].sum() * T.sum() / T[:, 0].sum() / T[:, 1].sum()
    p = 1 - chi2.cdf(x=c2, df=1)
    return p
df = pd.read_csv('advertisement_clicks.csv')
a = df[df['advertisement_id'] == 'A']
b = df[df['advertisement_id'] == 'B']
a = a['action']
b = b['action']
A_clk = a.sum()
A_noclk = a.size - a.sum()
B_clk = b.sum()
B_noclk = b.size - b.sum()
T = np.array([[A_clk, A_noclk], [B_clk, B_noclk]])
print(get_p_value(T))