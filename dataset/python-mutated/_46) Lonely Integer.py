import math
import os
import random
import re
import sys

def lonelyinteger(a):
    if False:
        for i in range(10):
            print('nop')
    for i in a:
        if a.count(i) == 1:
            return i
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    a = list(map(int, input().rstrip().split()))
    result = lonelyinteger(a)
    fptr.write(str(result) + '\n')
    fptr.close()