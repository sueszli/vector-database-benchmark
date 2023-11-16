import math
import os
import random
import re
import sys

def dynamicArray(n, queries):
    if False:
        while True:
            i = 10
    arr = [[] for _ in range(n)]
    lastAnswer = 0
    answers = []
    for i in range(len(queries)):
        if queries[i][0] == 1:
            (x, y) = (queries[i][1], queries[i][2])
            idx = (x ^ lastAnswer) % n
            arr[idx].append(y)
        else:
            (x, y) = (queries[i][1], queries[i][2])
            idx = (x ^ lastAnswer) % n
            lastAnswer = arr[idx][y % len(arr[idx])]
            answers.append(lastAnswer)
    return answers
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = int(first_multiple_input[0])
    q = int(first_multiple_input[1])
    queries = []
    for _ in range(q):
        queries.append(list(map(int, input().rstrip().split())))
    result = dynamicArray(n, queries)
    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')
    fptr.close()