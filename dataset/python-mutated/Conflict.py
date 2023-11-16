def conflict():
    if False:
        i = 10
        return i + 15
    raise AssertionError('Should not be executed')