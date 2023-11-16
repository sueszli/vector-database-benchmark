class Dummy(BaseException):
    pass

class GoodException(BaseException):

    def __new__(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        print('GoodException __new__')
        return Dummy(*args, **kwargs)

class BadException(BaseException):

    def __new__(cls, *args, **kwargs):
        if False:
            return 10
        print('BadException __new__')
        return 1
try:
    raise GoodException('good message')
except BaseException as good:
    print(type(good), good.args[0])
try:
    raise BadException('bad message')
except Exception as bad:
    print(type(bad), bad.args[0])
try:

    def gen():
        if False:
            for i in range(10):
                print('nop')
        yield
    gen().throw(BadException)
except Exception as genbad:
    print(type(genbad), genbad.args[0])