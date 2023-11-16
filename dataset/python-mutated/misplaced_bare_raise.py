try:
    pass
except Exception:
    raise
try:
    pass
except Exception:
    if True:
        raise
try:
    raise Exception
except Exception:
    try:
        raise Exception
    finally:
        raise

class ContextManager:

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, *args):
        if False:
            while True:
                i = 10
        raise
try:
    raise
except Exception:
    pass

def f():
    if False:
        i = 10
        return i + 15
    try:
        raise
    except Exception:
        pass

def g():
    if False:
        print('Hello World!')
    raise

def h():
    if False:
        while True:
            i = 10
    try:
        if True:

            def i():
                if False:
                    return 10
                raise
    except Exception:
        pass
    raise
raise
try:
    pass
except:

    def i():
        if False:
            print('Hello World!')
        raise
try:
    pass
except:

    class C:
        raise
try:
    pass
except:
    pass
finally:
    raise