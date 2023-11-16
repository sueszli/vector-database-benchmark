def redef(value):
    if False:
        i = 10
        return i + 15
    match value:
        case True:

            def fun(x, y):
                if False:
                    return 10
                return x
        case False:

            def fun(x, y):
                if False:
                    i = 10
                    return i + 15
                return y
    return fun