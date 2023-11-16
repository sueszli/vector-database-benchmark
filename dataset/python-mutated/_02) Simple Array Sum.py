import math
import os
import random
import re
import sys

def simpleArraySum(ar):
    if False:
        for i in range(10):
            print('nop')
    return sum(ar)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    ar_count = int(input().strip())
    ar = list(map(int, input().rstrip().split()))
    result = simpleArraySum(ar)
    fptr.write(str(result) + '\n')
    fptr.close()