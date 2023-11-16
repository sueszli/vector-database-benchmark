from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.mosaicplot import mosaic
data = {'a': 10, 'b': 15, 'c': 16}
mosaic(data, title='basic dictionary')
plt.show()
data = {('a', 'b'): 1, ('a', 'c'): 2, ('d', 'b'): 3, ('d', 'c'): 4}
mosaic(data, gap=0.05, title='complete dictionary')
plt.show()
rand = np.random.random
tuples = list(product(['bar', 'baz', 'foo', 'qux'], ['one', 'two']))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
data = pd.Series(rand(8), index=index)
mosaic(data, title='hierarchical index series')
plt.show()
rand = np.random.random
data = 1 + rand((2, 2))
mosaic(data, title='random non-labeled array')
plt.show()

def props(key):
    if False:
        i = 10
        return i + 15
    return {'color': 'r' if 'a' in key else 'gray'}

def labelizer(key):
    if False:
        for i in range(10):
            print('nop')
    return {('a',): 'first', ('b',): 'second', ('c',): 'third'}[key]
data = {'a': 10, 'b': 15, 'c': 16}
mosaic(data, title='colored dictionary', properties=props, labelizer=labelizer)
plt.show()
gender = ['male', 'male', 'male', 'female', 'female', 'female']
pet = ['cat', 'dog', 'dog', 'cat', 'dog', 'cat']
data = pd.DataFrame({'gender': gender, 'pet': pet})
mosaic(data, ['pet', 'gender'], title='DataFrame as Source')
plt.show()