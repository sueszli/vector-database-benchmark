import math
import os
import random
import re
import sys

def hourglassSum(arr):
    if False:
        for i in range(10):
            print('nop')
    SUM = []
    for i in range(len(arr) - 2):
        for j in range(len(arr) - 2):
            SUM.append(arr[i][j] + arr[i][j + 1] + arr[i][j + 2] + arr[i + 1][j + 1] + arr[i + 2][j] + arr[i + 2][j + 1] + arr[i + 2][j + 2])
    return max(SUM)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    arr = []
    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))
    result = hourglassSum(arr)
    fptr.write(str(result) + '\n')
    fptr.close()