import math
import os
import random
import re
import sys

def marcsCakewalk(calorie):
    if False:
        i = 10
        return i + 15
    calorie.sort(reverse=True)
    answer = 0
    for i in range(len(calorie)):
        answer += 2 ** i * calorie[i]
    return answer
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    calorie = list(map(int, input().rstrip().split()))
    result = marcsCakewalk(calorie)
    fptr.write(str(result) + '\n')
    fptr.close()