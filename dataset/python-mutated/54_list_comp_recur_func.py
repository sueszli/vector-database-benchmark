def recur1(a):
    if False:
        i = 10
        return i + 15
    return [recur1(b) for b in a]