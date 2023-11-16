def f():
    if False:
        while True:
            i = 10
    try:
        raise TypeError
    finally:
        print(1)
        try:
            raise ValueError
        finally:
            return 42
print(f())

def f():
    if False:
        return 10
    try:
        raise ValueError
    finally:
        print(1)
        try:
            raise TypeError
        finally:
            print(2)
            try:
                pass
            finally:
                return 42
print(f())

def f():
    if False:
        for i in range(10):
            print('nop')
    try:
        raise ValueError
    finally:
        print(1)
        try:
            raise TypeError
        finally:
            print(2)
            try:
                raise Exception
            finally:
                print(3)
                return 42
print(f())

def f():
    if False:
        while True:
            i = 10
    try:
        try:
            pass
        finally:
            print(2)
            return 42
    finally:
        print(1)
print(f())

def f():
    if False:
        while True:
            i = 10
    try:
        try:
            raise ValueError
        finally:
            print(2)
            return 42
    finally:
        print(1)
print(f())

def f():
    if False:
        i = 10
        return i + 15
    try:
        raise Exception
    finally:
        print(1)
        try:
            try:
                pass
            finally:
                print(3)
                return 42
        finally:
            print(2)
print(f())

def f():
    if False:
        i = 10
        return i + 15
    try:
        raise Exception
    finally:
        print(1)
        try:
            try:
                raise Exception
            finally:
                print(3)
                return 42
        finally:
            print(2)
print(f())