import math
import os
import random
import re
import sys

def birthday(s, d, m):
    if False:
        return 10
    possible_count = 0
    for i in range(len(s)):
        SUM = 0
        count = 0
        for j in range(i, len(s)):
            SUM += s[j]
            count += 1
            if SUM > d:
                break
            if count == m and SUM == d:
                possible_count += 1
                break
    return possible_count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    s = list(map(int, input().rstrip().split()))
    dm = input().rstrip().split()
    d = int(dm[0])
    m = int(dm[1])
    result = birthday(s, d, m)
    fptr.write(str(result) + '\n')
    fptr.close()