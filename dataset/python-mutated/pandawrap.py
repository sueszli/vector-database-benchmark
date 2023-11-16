__author__ = 'breddels'
__author__ = 'breddels'
import sys
import math
import pandas as pd
import vaex.dataset
import psutil

def meminfo():
    if False:
        i = 10
        return i + 15
    vmem = psutil.virtual_memory()
    print(('total mem', vmem.total / 1024.0 ** 3, 'avail', vmem.available / 1024.0 ** 3))

def test_pandas(dataset):
    if False:
        while True:
            i = 10
    meminfo()
    index = dataset.columns['random_index']
    x = pd.Series(dataset.columns['x'], index=index)
    y = pd.Series(dataset.columns['y'], index=index)
    z = pd.Series(dataset.columns['z'], index=index)
    f = pd.DataFrame({'x': x, 'y': y, 'z': z})
    print(f.x.mean())
    print(f.y.mean())
    print(f.z.mean())
    meminfo()
if __name__ == '__main__':
    input = sys.argv[1]
    dataset_in = vaex.dataset.load_file(input)
    test_pandas(dataset_in)