import math
import os
import random
import re
import sys

def miniMaxSum(arr):
    if False:
        print('Hello World!')
    minsum = sum(arr) - max(arr)
    maxsum = sum(arr) - min(arr)
    print(minsum, maxsum)
    return
if __name__ == '__main__':
    arr = list(map(int, input().rstrip().split()))
    miniMaxSum(arr)