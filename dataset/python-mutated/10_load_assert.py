def compact_traceback():
    if False:
        i = 10
        return i + 15
    tb = 5
    if not tb:
        raise AssertionError('traceback does not exist')