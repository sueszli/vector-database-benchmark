import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    if False:
        for i in range(10):
            print('nop')
    lastElement = arr[-1]
    for i in range(len(arr) - 1, -1, -1):
        if arr[i - 1] >= lastElement and i != 0:
            arr[i] = arr[i - 1]
            print(*arr)
        else:
            arr[i] = lastElement
            print(*arr)
            return
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)