import time
import numpy as np
import pandas as pd
if 'profile' not in dir():

    def profile(fn):
        if False:
            return 10
        return fn
SIZE = 10000000

@profile
def get_mean_for_indicator_poor(df, indicator):
    if False:
        for i in range(10):
            print('nop')
    gpby = df.groupby('indicator')
    means = gpby.mean()
    means_for_ind = means.loc[indicator]
    total = means_for_ind.sum()
    return total

@profile
def get_mean_for_indicator_better(df, indicator, rnd_cols):
    if False:
        while True:
            i = 10
    df_sub = df.query('indicator==@indicator')[rnd_cols]
    means_for_ind = df_sub.mean()
    total = means_for_ind.sum()
    return total

@profile
def run():
    if False:
        i = 10
        return i + 15
    arr = np.random.random((SIZE, 10))
    print(f'{arr.shape} shape for our array')
    df = pd.DataFrame(arr)
    rnd_cols = [f'c_{n}' for n in df.columns]
    df.columns = rnd_cols
    df2 = pd.DataFrame({'indicator': np.random.randint(0, 10, SIZE)})
    df = pd.concat((df2, df), axis=1)
    print('Head of our df:')
    print(df.head())
    print('Print results to check that we get the result')
    indicator = 2
    print(f'Mean for indicator {indicator} on better implementation {get_mean_for_indicator_better(df, indicator, rnd_cols):0.5f}')
    print(f'Mean for indicator {indicator} on poor implementation: {get_mean_for_indicator_poor(df, indicator):0.5f}')
if __name__ == '__main__':
    run()