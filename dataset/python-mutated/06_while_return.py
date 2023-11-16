def initiate_send(a, b, c, num_sent):
    if False:
        while True:
            i = 10
    while a and b:
        try:
            1 / (b - 1)
        except ZeroDivisionError:
            return 1
        if num_sent:
            c = 2
        return c

def initiate_send2(a, b):
    if False:
        i = 10
        return i + 15
    while a and b:
        try:
            1 / (b - 1)
        except ZeroDivisionError:
            return 1
        return 2
assert initiate_send(1, 1, 2, False) == 1
assert initiate_send(1, 2, 3, False) == 3
assert initiate_send(1, 2, 3, True) == 2
assert initiate_send2(1, 1) == 1
assert initiate_send2(1, 2) == 2