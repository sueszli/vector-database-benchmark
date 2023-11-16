import math
import os
import random
import re
import sys

def camelcase(s):
    if False:
        return 10
    count = 0
    for i in s:
        if i.isupper():
            count += 1
    return count + 1
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = camelcase(s)
    fptr.write(str(result) + '\n')
    fptr.close()