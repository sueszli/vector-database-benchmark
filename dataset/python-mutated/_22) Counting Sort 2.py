import math
import os
import random
import re
import sys

def countingSort(arr):
    if False:
        return 10
    counts = [0] * 100
    for i in arr:
        counts[i] += 1
    answer = []
    for i in range(len(counts)):
        while counts[i] > 0:
            counts[i] -= 1
            answer.append(i)
    return answer
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    result = countingSort(arr)
    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')
    fptr.close()