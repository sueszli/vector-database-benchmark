while 1:
    if __file__:
        while 1:
            if __file__:
                break
            raise RuntimeError
    else:
        raise RuntimeError

def _parseparam(s, end):
    if False:
        return 10
    while end > 0 and s.count(''):
        end = s.find(';')