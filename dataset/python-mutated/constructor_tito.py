from integration_test.taint import source, sink

class ParentWithConstructor:

    def __init__(self, arg):
        if False:
            print('Hello World!')
        ...

class ChildWithParentConstructor(ParentWithConstructor):

    def __init__(self, arg):
        if False:
            print('Hello World!')
        super(ChildWithParentConstructor, self).__init__(arg)

class ParentWithoutConstructor:
    ...

class ChildWithoutParentConstructor(ParentWithoutConstructor):

    def __init__(self, arg):
        if False:
            for i in range(10):
                print('nop')
        super(ChildWithoutParentConstructor, self).__init__(arg)

def test1():
    if False:
        for i in range(10):
            print('nop')
    tainted = source()
    child = ChildWithParentConstructor(tainted)
    sink(child.arg)

def test2():
    if False:
        print('Hello World!')
    tainted = source()
    child = ChildWithoutParentConstructor(tainted)
    sink(child.arg)