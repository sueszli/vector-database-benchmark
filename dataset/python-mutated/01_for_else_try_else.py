"""This program is self-checking!"""

def scan(items):
    if False:
        for i in range(10):
            print('nop')
    for i in items:
        try:
            5 / i
        except:
            continue
        else:
            break
    else:
        return 2
    return i
assert scan((0, 1)) == 1
assert scan((0, 0)) == 2
assert scan((3, 2, 1)) == 3