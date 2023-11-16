def taint_test(param1):
    if False:
        while True:
            i = 10
    x = tainted
    sink(x)
    sink(*x)
    sink(**x)