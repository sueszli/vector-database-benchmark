import math
import os
import random
import re
import sys

def stringConstruction(s):
    if False:
        while True:
            i = 10
    return len(set(s))
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(input().strip())
    for q_itr in range(q):
        s = input()
        result = stringConstruction(s)
        fptr.write(str(result) + '\n')
    fptr.close()