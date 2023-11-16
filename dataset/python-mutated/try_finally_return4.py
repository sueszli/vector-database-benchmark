def f(x):
    if False:
        i = 10
        return i + 15
    try:
        try:
            if x:
                return 42
        finally:
            try:
                print(1)
            finally:
                print(2)
            print(3)
        print(4)
    finally:
        print(5)
print(f(0))
print(f(1))

def f(x):
    if False:
        return 10
    try:
        try:
            if x:
                return 42
        finally:
            try:
                print(1)
                return 43
            finally:
                print(2)
            print(3)
        print(4)
    finally:
        print(5)
print(f(0))
print(f(1))

def f(x):
    if False:
        for i in range(10):
            print('nop')
    try:
        try:
            if x:
                return 42
        finally:
            try:
                print(1)
                raise ValueError
            finally:
                print(2)
            print(3)
        print(4)
    finally:
        print(5)
try:
    print(f(0))
except:
    print('caught')
try:
    print(f(1))
except:
    print('caught')

def f(x):
    if False:
        print('Hello World!')
    try:
        try:
            if x:
                return 42
        finally:
            try:
                print(1)
                raise Exception
            except:
                print(2)
            print(3)
        print(4)
    finally:
        print(5)
print(f(0))
print(f(1))