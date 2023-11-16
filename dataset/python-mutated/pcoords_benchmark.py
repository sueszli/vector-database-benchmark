import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from yellowbrick.features import ParallelCoordinates
import pandas as pd
import numpy as np

def plot_speedup(trials=5, factors=np.arange(1, 11)):
    if False:
        i = 10
        return i + 15

    def pcoords_time(X, y, fast=True):
        if False:
            for i in range(10):
                print('nop')
        (_, ax) = plt.subplots()
        oz = ParallelCoordinates(fast=fast, ax=ax)
        start = time.time()
        oz.fit_transform(X, y)
        delta = time.time() - start
        plt.cla()
        plt.clf()
        plt.close('all')
        return delta

    def pcoords_speedup(X, y):
        if False:
            print('Hello World!')
        fast_time = pcoords_time(X, y, fast=True)
        slow_time = pcoords_time(X, y, fast=False)
        return slow_time / fast_time
    data = load_iris()
    speedups = []
    variance = []
    for factor in factors:
        X = np.repeat(data.data, factor, axis=0)
        y = np.repeat(data.target, factor, axis=0)
        local_speedups = []
        for trial in range(trials):
            local_speedups.append(pcoords_speedup(X, y))
        local_speedups = np.array(local_speedups)
        speedups.append(local_speedups.mean())
        variance.append(local_speedups.std())
    speedups = np.array(speedups)
    variance = np.array(variance)
    series = pd.Series(speedups, index=factors)
    (_, ax) = plt.subplots(figsize=(9, 6))
    series.plot(ax=ax, marker='o', label='speedup factor', color='b')
    ax.fill_between(factors, speedups - variance, speedups + variance, alpha=0.25, color='b')
    ax.set_ylabel('speedup factor')
    ax.set_xlabel('dataset size (number of repeats in Iris dataset)')
    ax.set_title('Speed Improvement of Fast Parallel Coordinates')
    plt.savefig('images/fast_parallel_coordinates_speedup_benchmark.png')
if __name__ == '__main__':
    plot_speedup()