"""
python_magic_methods.py by xianhu
"""

class People(object):

    def __init__(self, name, age):
        if False:
            return 10
        self.name = name
        self.age = age
        return

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.name + ':' + str(self.age)

    def __lt__(self, other):
        if False:
            print('Hello World!')
        return self.name < other.name if self.name != other.name else self.age < other.age
print('\t'.join([str(item) for item in sorted([People('abc', 18), People('abe', 19), People('abe', 12), People('abc', 17)])]))

class MyDict(dict):

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        print('setitem:', key, value, self)
        super().__setitem__(key, value)
        return

    def __getitem__(self, item):
        if False:
            print('Hello World!')
        print('getitem:', item, self)
        if item not in self:
            temp = MyDict()
            super().__setitem__(item, temp)
            return temp
        return super().__getitem__(item)
test = MyDict()
test[0] = 'test'
test[1][2] = 'test1'
test[3][4][5] = 'test2'
print('==========================')