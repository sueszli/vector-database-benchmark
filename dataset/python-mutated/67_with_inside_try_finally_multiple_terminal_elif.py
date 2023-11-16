def foo():
    if False:
        return 10
    try:
        with x:
            if y:
                pass
            elif z:
                return z
            if y:
                pass
            elif z:
                return z
    finally:
        pass