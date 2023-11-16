import sys
if sys.version_info < (3, 0):
    print('py2')
else:
    print('py3')
if sys.version_info < (3, 0):
    if True:
        print('py2!')
    else:
        print('???')
else:
    print('py3')
if sys.version_info < (3, 0):
    print('PY2!')
else:
    print('PY3!')
if True:
    if sys.version_info < (3, 0):
        print('PY2')
    else:
        print('PY3')
if sys.version_info < (3, 0):
    print(1 if True else 3)
else:
    print('py3')
if sys.version_info < (3, 0):

    def f():
        if False:
            return 10
        print('py2')
else:

    def f():
        if False:
            return 10
        print('py3')
        print('This the next')
if sys.version_info > (3, 0):
    print('py3')
else:
    print('py2')
x = 1
if sys.version_info > (3, 0):
    print('py3')
else:
    print('py2')
x = 1
if sys.version_info > (3, 0):
    print('py3')
else:
    print('py2')
if sys.version_info > (3,):
    print('py3')
else:
    print('py2')
if True:
    if sys.version_info > (3,):
        print('py3')
    else:
        print('py2')
if sys.version_info < (3,):
    print('py2')
else:
    print('py3')

def f():
    if False:
        while True:
            i = 10
    if sys.version_info < (3, 0):
        try:
            yield
        finally:
            pass
    else:
        yield

class C:

    def g():
        if False:
            while True:
                i = 10
        pass
    if sys.version_info < (3, 0):

        def f(py2):
            if False:
                return 10
            pass
    else:

        def f(py3):
            if False:
                return 10
            pass

    def h():
        if False:
            for i in range(10):
                print('nop')
        pass
if True:
    if sys.version_info < (3, 0):
        2
    else:
        3
if sys.version_info < (3, 0):

    def f():
        if False:
            print('Hello World!')
        print('py2')

    def g():
        if False:
            for i in range(10):
                print('nop')
        print('py2')
else:

    def f():
        if False:
            return 10
        print('py3')

    def g():
        if False:
            return 10
        print('py3')
if True:
    if sys.version_info > (3,):
        print(3)
    print(2 + 3)
if True:
    if sys.version_info > (3,):
        print(3)
if True:
    if sys.version_info > (3,):
        print(3)
if True:
    if sys.version_info <= (3, 0):
        expected_error = []
    else:
        expected_error = ['<stdin>:1:5: Generator expression must be parenthesized', 'max(1 for i in range(10), key=lambda x: x+1)', '    ^']
if sys.version_info <= (3, 0):
    expected_error = []
else:
    expected_error = ['<stdin>:1:5: Generator expression must be parenthesized', 'max(1 for i in range(10), key=lambda x: x+1)', '    ^']
if sys.version_info > (3, 0):
    'this\nis valid'
    'the indentation on\n    this line is significant'
    'this isallowed too'
    'so isthis for some reason'
if sys.version_info > (3, 0):
    expected_error = []
if sys.version_info > (3, 0):
    expected_error = []
if sys.version_info > (3, 0):
    expected_error = []
if True:
    if sys.version_info > (3, 0):
        expected_error = []
if True:
    if sys.version_info > (3, 0):
        expected_error = []
if True:
    if sys.version_info > (3, 0):
        expected_error = []
if sys.version_info < (3, 12):
    print('py3')
if sys.version_info <= (3, 12):
    print('py3')
if sys.version_info <= (3, 12):
    print('py3')
if sys.version_info == 10000000:
    print('py3')
if sys.version_info < (3, 10000000):
    print('py3')
if sys.version_info <= (3, 10000000):
    print('py3')
if sys.version_info > (3, 12):
    print('py3')
if sys.version_info >= (3, 12):
    print('py3')
if sys.version_info[:2] >= (3, 0):
    print('py3')
if sys.version_info[:3] >= (3, 0):
    print('py3')
if sys.version_info[:2] > (3, 13):
    print('py3')
if sys.version_info[:3] > (3, 13):
    print('py3')