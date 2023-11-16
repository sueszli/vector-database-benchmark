def test():
    if False:
        return 10
    foo('128.0.0.1')
    foo('this is not an IP')
    foo('neither this')