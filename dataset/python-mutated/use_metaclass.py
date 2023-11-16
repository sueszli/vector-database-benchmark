class ListMetaclass(type):

    def __new__(cls, name, bases, attrs):
        if False:
            i = 10
            return i + 15
        attrs['add'] = lambda self, value: self.append(value)
        return type.__new__(cls, name, bases, attrs)

class MyList(list, metaclass=ListMetaclass):
    pass
L = MyList()
L.add(1)
L.add(2)
L.add(3)
L.add('END')
print(L)