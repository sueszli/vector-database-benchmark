import http.client

def div(a, b):
    if False:
        print('Hello World!')
    try:
        return a / b
    except ZeroDivisionError as exc:
        return None

class MyClass(object, metaclass=type):
    pass