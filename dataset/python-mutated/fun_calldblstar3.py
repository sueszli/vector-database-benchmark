def foo(**kw):
    if False:
        while True:
            i = 10
    print(sorted(kw.items()))

class Mapping:

    def keys(self):
        if False:
            print('Hello World!')
        return ['a', 'b', 'c', 'abcdefghijklmnopqrst']

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        if key == 'a':
            return 1
        else:
            return 2
foo(**Mapping())