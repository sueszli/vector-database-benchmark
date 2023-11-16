def f(b, *args, **kwargs):
    if False:
        return 10
    return (type(args), type(kwargs))