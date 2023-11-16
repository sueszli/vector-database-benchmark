def foo():
    if False:
        while True:
            i = 10
    seq = [1, 2, 3]
    v = 100
    i = 5
    while i > 0:
        print(i)
        for a in seq:
            if a == 2:
                break
        i -= 1
foo()

def bar():
    if False:
        return 10
    l = [1, 2, 3]
    for e1 in l:
        print(e1)
        for e2 in l:
            print(e1, e2)
            if e2 == 2:
                break
bar()