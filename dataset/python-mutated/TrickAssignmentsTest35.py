def someFunctionThatReturnsDeletedValueViaMaxtrixMult():
    if False:
        for i in range(10):
            print('nop')

    class C:

        def __matmul__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
    c = C()
    a = 1
    c @ 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaMaxtrixMult()
except UnboundLocalError:
    print('OK, object matrix mult correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInplaceMaxtrixMult():
    if False:
        print('Hello World!')

    class C:

        def __imatmul__(self, other):
            if False:
                return 10
            nonlocal a
            del a
    c = C()
    a = 1
    c @= 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaInplaceMaxtrixMult()
except UnboundLocalError:
    print('OK, object inplace matrix mult correctly deleted an item.')
else:
    print('Ouch.!')