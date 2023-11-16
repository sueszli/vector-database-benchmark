import math
import os
import random
import re
import sys

def repeatedString(s, n):
    if False:
        return 10
    return s.count('a') * (n // len(s)) + s[:n % len(s)].count('a')
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    n = int(input())
    result = repeatedString(s, n)
    fptr.write(str(result) + '\n')
    fptr.close()