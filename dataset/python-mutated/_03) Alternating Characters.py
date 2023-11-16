import math
import os
import random
import re
import sys

def alternatingCharacters(s):
    if False:
        print('Hello World!')
    count = 0
    for i in range(len(s) - 1):
        if s[i] == 'A' and s[i + 1] == 'B' or (s[i] == 'B' and s[i + 1] == 'A'):
            continue
        count += 1
    return count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(input())
    for q_itr in range(q):
        s = input()
        result = alternatingCharacters(s)
        fptr.write(str(result) + '\n')
    fptr.close()