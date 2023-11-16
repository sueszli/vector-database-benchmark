import math
import os
import random
import re
import sys

def theLoveLetterMystery(s):
    if False:
        while True:
            i = 10
    answer = 0
    for i in range(len(s) // 2):
        diff = abs(ord(s[i]) - ord(s[len(s) - i - 1]))
        answer += diff
    return answer
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(input().strip())
    for q_itr in range(q):
        s = input()
        result = theLoveLetterMystery(s)
        fptr.write(str(result) + '\n')
    fptr.close()