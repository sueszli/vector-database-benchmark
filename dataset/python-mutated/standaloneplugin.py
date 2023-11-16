class SamplePlugin(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.__count = 10

    def do_work(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__count

class FooPlugin(object):
    """
    Some class that doesn't implement the specified plugin interface.
    """

    def foo():
        if False:
            i = 10
            return i + 15
        pass