"""
This is an example implementation of ALS for learning how to use Spark. Please refer to
pyspark.ml.recommendation.ALS for more conventional use.

This example requires numpy (http://www.numpy.org/)
"""
import sys
import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark.sql import SparkSession
LAMBDA = 0.01
np.random.seed(42)

def rmse(R: np.ndarray, ms: np.ndarray, us: np.ndarray) -> np.float64:
    if False:
        while True:
            i = 10
    diff = R - ms * us.T
    return np.sqrt(np.sum(np.power(diff, 2)) / (M * U))

def update(i: int, mat: np.ndarray, ratings: np.ndarray) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    uu = mat.shape[0]
    ff = mat.shape[1]
    XtX = mat.T * mat
    Xty = mat.T * ratings[i, :].T
    for j in range(ff):
        XtX[j, j] += LAMBDA * uu
    return np.linalg.solve(XtX, Xty)
if __name__ == '__main__':
    '\n    Usage: als [M] [U] [F] [iterations] [partitions]"\n    '
    print('WARN: This is a naive implementation of ALS and is given as an\n      example. Please use pyspark.ml.recommendation.ALS for more\n      conventional use.', file=sys.stderr)
    spark = SparkSession.builder.appName('PythonALS').getOrCreate()
    sc = spark.sparkContext
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    U = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    F = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    ITERATIONS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    partitions = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    print('Running ALS with M=%d, U=%d, F=%d, iters=%d, partitions=%d\n' % (M, U, F, ITERATIONS, partitions))
    R = matrix(rand(M, F)) * matrix(rand(U, F).T)
    ms: matrix = matrix(rand(M, F))
    us: matrix = matrix(rand(U, F))
    Rb = sc.broadcast(R)
    msb = sc.broadcast(ms)
    usb = sc.broadcast(us)
    for i in range(ITERATIONS):
        ms_ = sc.parallelize(range(M), partitions).map(lambda x: update(x, usb.value, Rb.value)).collect()
        ms = matrix(np.array(ms_)[:, :, 0])
        msb = sc.broadcast(ms)
        us_ = sc.parallelize(range(U), partitions).map(lambda x: update(x, msb.value, Rb.value.T)).collect()
        us = matrix(np.array(us_)[:, :, 0])
        usb = sc.broadcast(us)
        error = rmse(R, ms, us)
        print('Iteration %d:' % i)
        print('\nRMSE: %5.4f\n' % error)
    spark.stop()