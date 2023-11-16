import os
import sys

def timeConversion(s):
    if False:
        for i in range(10):
            print('nop')
    if s[-2:].upper() == 'AM' and s[:2] != '12':
        time = s[:-2]
    elif s[-2:].upper() == 'AM' and s[:2] == '12':
        s = '00' + s[2:]
        time = s[:-2]
    if s[-2:].upper() == 'PM' and s[:2] != '12':
        time = int(s[0:2]) + 12
        time = str(time) + s[2:-2]
    elif s[-2:].upper() == 'PM' and s[:2] == '12':
        time = int(s[0:2])
        time = str(time) + s[2:-2]
    return time
if __name__ == '__main__':
    f = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = timeConversion(s)
    f.write(result + '\n')
    f.close()