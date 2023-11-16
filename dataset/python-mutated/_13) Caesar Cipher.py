import math
import os
import random
import re
import sys

def caesarCipher(s, k):
    if False:
        return 10
    k %= 26
    newString = ''
    for i in s:
        asciiValue = ord(i)
        if asciiValue >= 97 and asciiValue <= 122:
            if asciiValue + k > 122:
                newString += chr(96 + asciiValue - 122 + k)
            else:
                newString += chr(asciiValue + k)
        elif asciiValue >= 65 and asciiValue <= 90:
            if asciiValue + k > 90:
                newString += chr(64 + asciiValue - 90 + k)
            else:
                newString += chr(asciiValue + k)
        else:
            newString += i
    return newString
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    s = input()
    k = int(input().strip())
    result = caesarCipher(s, k)
    fptr.write(result + '\n')
    fptr.close()