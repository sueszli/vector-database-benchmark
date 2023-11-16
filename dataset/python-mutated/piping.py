import sys

def read_all():
    if False:
        i = 10
        return i + 15
    fails = 0
    for line in sys.stdin:
        if 'FAIL' in line:
            fails += 1
    print("%d lines with 'FAIL' found!" % fails)

def read_some():
    if False:
        i = 10
        return i + 15
    for line in sys.stdin:
        if 'FAIL' in line:
            print("Line with 'FAIL' found!")
            sys.stdin.close()
            break

def read_none():
    if False:
        i = 10
        return i + 15
    sys.stdin.close()
globals()[sys.argv[1]]()