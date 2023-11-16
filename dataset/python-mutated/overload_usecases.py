def impl4(z, *args, kw=None):
    if False:
        while True:
            i = 10
    if z > 10:
        return 1
    else:
        return -1

def impl5(z, b, *args, kw=None):
    if False:
        return 10
    if z > 10:
        return 1
    else:
        return -1

def var_positional_impl(a, *star_args_token, kw=None, kw1=12):
    if False:
        for i in range(10):
            print('nop')

    def impl(a, b, f, kw=None, kw1=12):
        if False:
            i = 10
            return i + 15
        if a > 10:
            return 1
        else:
            return -1
    return impl