try:
    NativeBaseClass
except NameError:
    print('SKIP')
    raise SystemExit
n = NativeBaseClass(test='direct kwarg')
print('.test:', n.test)
n.test = 'test set directly'
print('.test:', n.test)

class A(NativeBaseClass):
    pass
a = A(test='subclass kwarg')
print('.test:', a.test)
a.test = 'test set indirectly'
print('.test:', a.test)
a.new_attribute = True
print('.new_attribute', a.new_attribute)
a.print_subclass_attr('new_attribute')
print(a[0])

class B(NativeBaseClass):

    def __init__(self, suffix):
        if False:
            i = 10
            return i + 15
        super().__init__(test='super init ' + suffix)
b = B('suffix')
print('.test:', b.test)
b.test = 'test set indirectly through b'
print('.test:', b.test)
b.new_attribute = 'hello'
print('.new_attribute', b.new_attribute)
b.print_subclass_attr('new_attribute')
print(b[0])