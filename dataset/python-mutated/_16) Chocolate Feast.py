import math
import os
import random
import re
import sys

def chocolateFeast(n, c, m):
    if False:
        i = 10
        return i + 15
    count = n // c
    x = count
    while x >= m:
        (a, b) = divmod(x, m)
        count += a
        x = a + b
    return count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input().strip())
    for t_itr in range(t):
        first_multiple_input = input().rstrip().split()
        n = int(first_multiple_input[0])
        c = int(first_multiple_input[1])
        m = int(first_multiple_input[2])
        result = chocolateFeast(n, c, m)
        fptr.write(str(result) + '\n')
    fptr.close()