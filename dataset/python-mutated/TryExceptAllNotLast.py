def f():
    if False:
        i = 10
        return i + 15
    try:
        raise A
    except:
        print('caught')
    except A:
        print('hit')