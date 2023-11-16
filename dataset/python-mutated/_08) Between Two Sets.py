import math
import os
import random
import re
import sys

def getTotalX(a, b):
    if False:
        return 10
    x = max(a)
    count = 0
    y = min(b)
    for i in range(x, y + 1):
        for j in range(len(a)):
            if not i % a[j] == 0:
                break
            if j == len(a) - 1:
                for k in range(len(b)):
                    if not b[k] % i == 0:
                        break
                    if k == len(b) - 1:
                        count += 1
    return count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = int(first_multiple_input[0])
    m = int(first_multiple_input[1])
    arr = list(map(int, input().rstrip().split()))
    brr = list(map(int, input().rstrip().split()))
    total = getTotalX(arr, brr)
    fptr.write(str(total) + '\n')
    fptr.close()