LIST = ['spam', 'eggs', 21]

class _Extended:
    list = LIST
    string = 'not a list'

    def __getitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        return LIST
EXTENDED = _Extended()

class _Iterable:

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(LIST)
ITERABLE = _Iterable()