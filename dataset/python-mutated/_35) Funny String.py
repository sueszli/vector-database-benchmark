import math
import os
import random
import re
import sys

def funnyString(s):
    if False:
        print('Hello World!')
    asciiValues = []
    for i in s:
        asciiValues.append(ord(i))
    asciiValuesReverse = []
    for i in range(len(asciiValues)):
        asciiValuesReverse.append(asciiValues[len(asciiValues) - i - 1])
    for i in range(len(s) - 1):
        if abs(asciiValues[i + 1] - asciiValues[i]) != abs(asciiValuesReverse[i + 1] - asciiValuesReverse[i]):
            return 'Not Funny'
    return 'Funny'
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    q = int(input().strip())
    for q_itr in range(q):
        s = input()
        result = funnyString(s)
        fptr.write(result + '\n')
    fptr.close()