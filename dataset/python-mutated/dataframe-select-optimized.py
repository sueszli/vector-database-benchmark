import pandas as pd
import numpy as np
import timeit
np.random.seed(1)
column_names_example = [i for i in range(10000)]
index = pd.MultiIndex.from_tuples([('left', c) for c in column_names_example] + [('right', c) for c in column_names_example])
df = pd.DataFrame(np.random.rand(1000, 20000), columns=index)

def keep_column(left_col, right_col):
    if False:
        i = 10
        return i + 15
    return left_col[left_col.first_valid_index()] > right_col[right_col.last_valid_index()]

def do_it_original():
    if False:
        i = 10
        return i + 15
    v = [c for c in column_names_example if keep_column(df['left'][c], df['right'][c])]
    return v

def do_it():
    if False:
        for i in range(10):
            print('nop')
    left_cols = df['left'].loc[:, column_names_example]
    right_cols = df['right'].loc[:, column_names_example]
    v = left_cols.columns[left_cols.iloc[0] > right_cols.iloc[-1]]
    return v
do_it()