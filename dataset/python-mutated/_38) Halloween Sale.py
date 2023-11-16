import math
import os
import random
import re
import sys

def howManyGames(p, d, m, s):
    if False:
        print('Hello World!')
    games = 0
    while s >= 0:
        s -= p
        p = max(p - d, m)
        games += 1
    return games - 1
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    p = int(first_multiple_input[0])
    d = int(first_multiple_input[1])
    m = int(first_multiple_input[2])
    s = int(first_multiple_input[3])
    answer = howManyGames(p, d, m, s)
    fptr.write(str(answer) + '\n')
    fptr.close()