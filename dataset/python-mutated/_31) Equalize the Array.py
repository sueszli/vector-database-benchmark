import math
import os
import random
import re
import sys

def equalizeArray(arr):
    if False:
        while True:
            i = 10
    return len(arr) - max([arr.count(i) for i in set(arr)])
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    result = equalizeArray(arr)
    fptr.write(str(result) + '\n')
    fptr.close()