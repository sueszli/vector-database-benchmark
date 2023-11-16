def f():
    if False:
        for i in range(10):
            print('nop')
    try:
        raise ValueError()
    finally:
        print('finally')
        return 0
    print('got here')
print(f())

def f():
    if False:
        print('Hello World!')
    try:
        try:
            raise ValueError
        finally:
            print('finally 1')
        print('got here')
    finally:
        print('finally 2')
        return 2
    print('got here')
print(f())

def f():
    if False:
        return 10
    try:
        try:
            raise ValueError
        finally:
            print('finally 1')
            return 1
        print('got here')
    finally:
        print('finally 2')
    print('got here')
print(f())

def f():
    if False:
        i = 10
        return i + 15
    try:
        try:
            raise ValueError
        finally:
            print('finally 1')
            return 1
        print('got here')
    finally:
        print('finally 2')
        return 2
    print('got here')
print(f())

def f():
    if False:
        while True:
            i = 10
    try:
        try:
            raise ValueError
        except:
            raise
        print('got here')
    finally:
        print('finally')
        return 0
    print('got here')
print(f())

def f():
    if False:
        print('Hello World!')
    try:
        try:
            try:
                raise ValueError
            except:
                raise
        except:
            raise
    finally:
        print('finally')
        return 0
print(f())

def f():
    if False:
        return 10
    try:
        raise ValueError
    except NonExistingError:
        pass
    finally:
        print('finally')
        return 0
print(f())

def f():
    if False:
        return 10
    try:
        raise ValueError
    finally:
        print('finally')
        return 0
print(f())