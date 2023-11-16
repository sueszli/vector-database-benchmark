my_dict = {}
my_var = 0

def foo():
    if False:
        print('Hello World!')
    my_var += 1

def bar():
    if False:
        return 10
    global my_var
    my_var += 1

def baz():
    if False:
        for i in range(10):
            print('nop')
    global my_var
    global my_dict
    my_dict[my_var] += 1

def dec(x):
    if False:
        while True:
            i = 10
    return x

@dec
def f():
    if False:
        return 10
    dec = 1
    return dec

class Class:

    def f(self):
        if False:
            print('Hello World!')
        print(my_var)
        my_var = 1

class Class:
    my_var = 0

    def f(self):
        if False:
            i = 10
            return i + 15
        print(my_var)
        my_var = 1
import sys

def main():
    if False:
        return 10
    print(sys.argv)
    try:
        3 / 0
    except ZeroDivisionError:
        import sys
        sys.exit(1)
import sys

def main():
    if False:
        return 10
    print(sys.argv)
    for sys in range(5):
        pass
import requests_mock as rm

def requests_mock(requests_mock: rm.Mocker):
    if False:
        for i in range(10):
            print('nop')
    print(rm.ANY)
import sklearn.base
import mlflow.sklearn

def f():
    if False:
        while True:
            i = 10
    import sklearn
    mlflow