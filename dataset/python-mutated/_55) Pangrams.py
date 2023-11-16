import math
import os
import random
import re
import sys

def pangrams(s):
    if False:
        while True:
            i = 10
    alphabets = [0] * 26
    for i in s:
        if i.isalpha():
            index = ord(i.lower()) - ord('a')
            alphabets[index] += 1
    for i in alphabets:
        if i == 0:
            return 'not pangram'
    return 'pangram'
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = pangrams(s)
    fptr.write(result + '\n')
    fptr.close()