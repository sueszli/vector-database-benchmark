import math
import os
import random
import re
import sys

def viralAdvertising(n):
    if False:
        for i in range(10):
            print('nop')
    shares = 5
    likes = math.floor(shares / 2)
    total = likes
    for i in range(n - 1):
        shares = likes * 3
        likes = math.floor(shares / 2)
        total += likes
    return total
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()