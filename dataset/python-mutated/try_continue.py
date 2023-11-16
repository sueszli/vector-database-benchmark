def f():
    if False:
        i = 10
        return i + 15
    lst = [1, 2, 3]
    for x in lst:
        print('a', x)
        try:
            if x == 2:
                raise Exception
        except Exception:
            continue
        print('b', x)
f()