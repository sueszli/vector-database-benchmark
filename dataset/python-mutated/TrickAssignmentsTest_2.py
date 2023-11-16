def someFunctionThatReturnsDeletedValueViaLong():
    if False:
        while True:
            i = 10

    class C:

        def __int__(self):
            if False:
                return 10
            a.append(2)
            return False
    c = C()
    a = [1]
    long(c)
    return a
if someFunctionThatReturnsDeletedValueViaLong()[-1] == 2:
    print('OK, object long correctly modified an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaNot():
    if False:
        for i in range(10):
            print('nop')

    class C:

        def __nonzero__(self):
            if False:
                for i in range(10):
                    print('nop')
            a.append(2)
            return False
    c = C()
    a = [1]
    not c
    return a
if someFunctionThatReturnsDeletedValueViaNot()[-1] == 2:
    print('OK, object bool correctly modified an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaCompare():
    if False:
        i = 10
        return i + 15

    class C:

        def __cmp__(self, other):
            if False:
                i = 10
                return i + 15
            a.append(2)
            return 0
    c = C()
    a = [1]
    c < None
    return a
if someFunctionThatReturnsDeletedValueViaCompare()[-1] == 2:
    print('OK, object compare correctly modified an item.')
else:
    print('Ouch.!')