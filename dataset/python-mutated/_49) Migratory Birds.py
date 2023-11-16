import math
import os
import random
import re
import sys

def migratoryBirds(arr):
    if False:
        while True:
            i = 10
    counts = {}
    for i in arr:
        if i in counts:
            counts[i] += 1
        else:
            counts[i] = 1
    print(counts)
    m = max(counts.values())
    minKey = 100000
    for (key, value) in counts.items():
        if value == m:
            if minKey > key:
                minKey = key
    return minKey
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    arr_count = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    result = migratoryBirds(arr)
    fptr.write(str(result) + '\n')
    fptr.close()