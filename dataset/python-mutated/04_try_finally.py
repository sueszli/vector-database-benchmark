def getvalue(self):
    if False:
        for i in range(10):
            print('nop')
    try:
        return 3
    finally:
        return 1

def getvalue1(self):
    if False:
        while True:
            i = 10
    try:
        return 4
    finally:
        pass
    return 2

def handle_read(self):
    if False:
        for i in range(10):
            print('nop')
    try:
        data = 5
    except ZeroDivisionError:
        return
    except OSError as why:
        return why
    return data

def __exit__(self, type, value, traceback):
    if False:
        while True:
            i = 10
    try:
        value()
    except StopIteration as exc:
        return exc
    except RuntimeError as exc:
        return exc
    return