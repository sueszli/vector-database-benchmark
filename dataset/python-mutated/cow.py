def Say(*args, **kwargs):
    if False:
        print('Hello World!')
    print(*args, **kwargs)

def Quiet():
    if False:
        i = 10
        return i + 15
    pass