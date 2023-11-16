import math
import os
import random
import re
import sys

def superReducedString(s):
    if False:
        while True:
            i = 10
    stringList = list(s)
    i = 0
    while i < len(stringList) - 1:
        if stringList[i] == stringList[i + 1]:
            del stringList[i]
            del stringList[i]
            i = 0
            if len(stringList) == 0:
                return 'Empty String'
        else:
            i += 1
    return ''.join(stringList)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = superReducedString(s)
    fptr.write(result + '\n')
    fptr.close()