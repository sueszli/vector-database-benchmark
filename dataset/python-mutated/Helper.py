def get_non_none(*args):
    if False:
        return 10
    for arg in args:
        if arg is not None:
            return arg