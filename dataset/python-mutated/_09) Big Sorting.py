import math
import os
import random
import re
import sys

def bigSorting(unsorted):
    if False:
        while True:
            i = 10
    unsorted.sort()
    return sorted(unsorted, key=len)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    unsorted = []
    for _ in range(n):
        unsorted_item = input()
        unsorted.append(unsorted_item)
    result = bigSorting(unsorted)
    fptr.write('\n'.join(result))
    fptr.write('\n')
    fptr.close()