import os
import sys

def getMoneySpent(keyboards, drives, b):
    if False:
        print('Hello World!')
    s = 0
    for i in keyboards:
        for j in drives:
            if i + j <= b and s < i + j:
                s = i + j
    if s == 0:
        return -1
    return s
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    bnm = input().split()
    b = int(bnm[0])
    n = int(bnm[1])
    m = int(bnm[2])
    keyboards = list(map(int, input().rstrip().split()))
    drives = list(map(int, input().rstrip().split()))
    moneySpent = getMoneySpent(keyboards, drives, b)
    fptr.write(str(moneySpent) + '\n')
    fptr.close()