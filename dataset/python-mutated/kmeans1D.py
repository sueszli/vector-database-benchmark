"""
kmeans1D.py
Author : domlysz@gmail.com
Date : february 2016
License : GPL

This file is part of BlenderGIS.
This is a kmeans implementation optimized for 1D data.

Original kmeans code :
https://gist.github.com/iandanforth/5862470

1D optimizations are inspired from this talking :
http://stats.stackexchange.com/questions/40454/determine-different-clusters-of-1d-data-from-database

Optimizations consists to :
-sort the data and initialize clusters with a quantile classification
-compute distance in 1D instead of euclidean
-optimize only the borders of the clusters instead of test each cluster values

Clustering results are similar to Jenks natural break and ckmeans algorithms.
There are Python implementations of these alg. based on javascript code from simple-statistics library :
* Jenks : https://gist.github.com/llimllib/4974446 (https://gist.github.com/tmcw/4977508)
* Ckmeans : https://github.com/llimllib/ckmeans (https://github.com/simple-statistics/simple-statistics/blob/master/src/ckmeans.js)

But both are terribly slow because there is a lot of exponential-time looping. These algorithms makes this somewhat inevitable.
In contrast, this script works in a reasonable time, but keep in mind it's not Jenks. We just use cluster's centroids (mean) as
reference to distribute the values while Jenks try to minimize within-class variance, and maximizes between group variance.
"""
from ..utils.timing import perf_clock

def kmeans1d(data, k, cutoff=False, maxIter=False):
    if False:
        return 10
    "\n\tCompute natural breaks of a one dimensionnal list through an optimized kmeans algorithm\n\tInputs:\n\t* data = input list, must be sorted beforehand\n\t* k = number of expected classes\n\t* cutoff (optional) = stop algorithm when centroids shift are under this value\n\t* maxIter (optional) = stop algorithm when iteration count reach this value\n\tOutput:\n\t* A list of k clusters. A cluster is represented by a tuple containing first and last index of the cluster's values.\n\tUse these index on the input data list to retreive the effectives values containing in a cluster.\n\t"

    def getClusterValues(cluster):
        if False:
            i = 10
            return i + 15
        (i, j) = cluster
        return data[i:j + 1]

    def getClusterCentroid(cluster):
        if False:
            print('Hello World!')
        values = getClusterValues(cluster)
        return sum(values) / len(values)
    n = len(data)
    if k >= n:
        raise ValueError('Too many expected classes')
    if k == 1:
        return [[0, n - 1]]
    q = int(n // k)
    if q == 1:
        raise ValueError('Too many expected classes')
    clusters = [[i, i + q - 1] for i in range(0, q * k, q)]
    clusters[-1][1] = n - 1
    centroids = [getClusterCentroid(c) for c in clusters]
    loopCounter = 0
    changeOccured = True
    while changeOccured:
        loopCounter += 1
        changeOccured = False
        for i in range(k - 1):
            c1 = clusters[i]
            c2 = clusters[i + 1]
            adjusted = False
            while True:
                if c1[0] == c1[1]:
                    break
                breakValue = data[c1[1]]
                dst1 = abs(breakValue - centroids[i])
                dst2 = abs(breakValue - centroids[i + 1])
                if dst1 > dst2:
                    c1[1] -= 1
                    c2[0] -= 1
                    adjusted = True
                else:
                    break
            if not adjusted:
                while True:
                    if c2[0] == c2[1]:
                        break
                    breakValue = data[c2[0]]
                    dst1 = abs(breakValue - centroids[i])
                    dst2 = abs(breakValue - centroids[i + 1])
                    if dst2 > dst1:
                        c2[0] += 1
                        c1[1] += 1
                        adjusted = True
                    else:
                        break
            if adjusted:
                changeOccured = True
        newCentroids = [getClusterCentroid(c) for c in clusters]
        biggest_shift = max([abs(newCentroids[i] - centroids[i]) for i in range(k)])
        centroids = newCentroids
        if cutoff and biggest_shift < cutoff or (maxIter and loopCounter == maxIter):
            break
    return clusters

def getClustersValues(data, clusters):
    if False:
        while True:
            i = 10
    return [data[i:j + 1] for (i, j) in clusters]

def getBreaks(data, clusters, includeBounds=False):
    if False:
        while True:
            i = 10
    if includeBounds:
        return [data[0]] + [data[j] for (i, j) in clusters]
    else:
        return [data[j] for (i, j) in clusters[:-1]]
if __name__ == '__main__':
    import random, time
    data = [random.uniform(0, 1000) for i in range(10000)]
    data.extend([random.uniform(2000, 4000) for i in range(10000)])
    data.sort()
    k = 4
    print('---------------')
    print('%i values, %i classes' % (len(data), k))
    t1 = perf_clock()
    clusters = kmeans1d(data, k)
    t2 = perf_clock()
    print('Completed in %f seconds' % (t2 - t1))
    print('Breaks :')
    print(getBreaks(data, clusters))
    print('Clusters details (nb values, min, max) :')
    for clusterValues in getClustersValues(data, clusters):
        print(len(clusterValues), clusterValues[0], clusterValues[-1])