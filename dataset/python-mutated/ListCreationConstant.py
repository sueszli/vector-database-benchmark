import itertools

def calledRepeatedly():
    if False:
        while True:
            i = 10
    l = [50, 51, 52, 53, 54, 55, 56, 57]
    l = 1
    return l
for x in itertools.repeat(None, 50000):
    calledRepeatedly()
print('OK.')