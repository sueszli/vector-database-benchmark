import math
import os
import random
import re
import sys
from collections import deque

def circularArrayRotation(a, k, queries):
    if False:
        return 10
    items = deque(a)
    items.rotate(k)
    ret_list = []
    for q in queries:
        ret_list.append(items[q])
    return ret_list
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nkq = input().split()
    n = int(nkq[0])
    k = int(nkq[1])
    q = int(nkq[2])
    a = list(map(int, input().rstrip().split()))
    queries = []
    for _ in range(q):
        queries_item = int(input())
        queries.append(queries_item)
    result = circularArrayRotation(a, k, queries)
    fptr.write('\n'.join(map(str, result)))
    fptr.write('\n')
    fptr.close()