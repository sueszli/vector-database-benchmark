import math
import os
import random
import re
import sys

def quickSort(arr):
    if False:
        i = 10
        return i + 15
    p = arr[0]
    left = [arr[i] for i in range(len(arr)) if arr[i] < p]
    equal = [arr[i] for i in range(len(arr)) if arr[i] == p]
    right = [arr[i] for i in range(len(arr)) if arr[i] > p]
    answer = left + equal + right
    return answer
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    result = quickSort(arr)
    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')
    fptr.close()