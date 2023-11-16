def someFunctionThatReturnsDeletedValueViaAttributeLookup():
    if False:
        while True:
            i = 10

    class C:

        def __getattr__(self, attr_name):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
    c = C()
    a = 1
    c.something
    return a
try:
    someFunctionThatReturnsDeletedValueViaAttributeLookup()
except UnboundLocalError:
    print('OK, object attribute look-up correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaAttributeSetting():
    if False:
        while True:
            i = 10

    class C:

        def __setattr__(self, attr_name, value):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
    c = C()
    a = 1
    c.something = 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaAttributeSetting()
except UnboundLocalError:
    print('OK, object attribute setting correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaAttributeDel():
    if False:
        print('Hello World!')

    class C:

        def __delattr__(self, attr_name):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
            return True
    c = C()
    a = 1
    del c.something
    return a
try:
    someFunctionThatReturnsDeletedValueViaAttributeDel()
except UnboundLocalError:
    print('OK, object attribute del correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaItemLookup():
    if False:
        while True:
            i = 10

    class C:

        def __getitem__(self, attr_name):
            if False:
                print('Hello World!')
            nonlocal a
            del a
    c = C()
    a = 1
    c[2]
    return a
try:
    someFunctionThatReturnsDeletedValueViaItemLookup()
except UnboundLocalError:
    print('OK, object subscript look-up correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaItemSetting():
    if False:
        print('Hello World!')

    class C:

        def __setitem__(self, attr_name, value):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
    c = C()
    a = 1
    c[2] = 3
    return a
try:
    someFunctionThatReturnsDeletedValueViaItemSetting()
except UnboundLocalError:
    print('OK, object subscript setting correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaItemDel():
    if False:
        return 10

    class C:

        def __delitem__(self, attr_name):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
    c = C()
    a = 1
    del c[2]
    return a
try:
    someFunctionThatReturnsDeletedValueViaItemDel()
except UnboundLocalError:
    print('OK, object subscript del correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaCall():
    if False:
        return 10

    class C:

        def __call__(self):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
    c = C()
    a = 1
    c()
    return a
try:
    someFunctionThatReturnsDeletedValueViaCall()
except UnboundLocalError:
    print('OK, object call correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaAdd():
    if False:
        return 10

    class C:

        def __add__(self, other):
            if False:
                print('Hello World!')
            nonlocal a
            del a
    c = C()
    a = 1
    c + 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaAdd()
except UnboundLocalError:
    print('OK, object add correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaSub():
    if False:
        print('Hello World!')

    class C:

        def __sub__(self, other):
            if False:
                print('Hello World!')
            nonlocal a
            del a
    c = C()
    a = 1
    c - 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaSub()
except UnboundLocalError:
    print('OK, object sub correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaMul():
    if False:
        while True:
            i = 10

    class C:

        def __mul__(self, other):
            if False:
                print('Hello World!')
            nonlocal a
            del a
            return 7
    c = C()
    a = 1
    c * 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaMul()
except UnboundLocalError:
    print('OK, object mul correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaRemainder():
    if False:
        i = 10
        return i + 15

    class C:

        def __mod__(self, other):
            if False:
                i = 10
                return i + 15
            nonlocal a
            del a
            return 7
    c = C()
    a = 1
    c % 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaRemainder()
except UnboundLocalError:
    print('OK, object remainder correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaDivmod():
    if False:
        print('Hello World!')

    class C:

        def __divmod__(self, other):
            if False:
                i = 10
                return i + 15
            nonlocal a
            del a
            return 7
    c = C()
    a = 1
    divmod(c, 1)
    return a
try:
    someFunctionThatReturnsDeletedValueViaDivmod()
except UnboundLocalError:
    print('OK, object divmod correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaPower():
    if False:
        while True:
            i = 10

    class C:

        def __pow__(self, other):
            if False:
                return 10
            nonlocal a
            del a
            return 7
    c = C()
    a = 1
    c ** 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaPower()
except UnboundLocalError:
    print('OK, object power correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaUnaryMinus():
    if False:
        i = 10
        return i + 15

    class C:

        def __neg__(self):
            if False:
                i = 10
                return i + 15
            nonlocal a
            del a
            return 7
    c = C()
    a = 1
    -c
    return a
try:
    someFunctionThatReturnsDeletedValueViaUnaryMinus()
except UnboundLocalError:
    print('OK, object unary minus correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaUnaryPlus():
    if False:
        return 10

    class C:

        def __pos__(self):
            if False:
                i = 10
                return i + 15
            nonlocal a
            del a
            return 7
    c = C()
    a = 1
    +c
    return a
try:
    someFunctionThatReturnsDeletedValueViaUnaryPlus()
except UnboundLocalError:
    print('OK, object unary plus correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaNot():
    if False:
        return 10

    class C:

        def __bool__(self):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    not c
    return a
try:
    someFunctionThatReturnsDeletedValueViaNot()
except UnboundLocalError:
    print('OK, object bool correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInvert():
    if False:
        while True:
            i = 10

    class C:

        def __invert__(self):
            if False:
                print('Hello World!')
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    ~c
    return a
try:
    someFunctionThatReturnsDeletedValueViaInvert()
except UnboundLocalError:
    print('OK, object invert correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaLshift():
    if False:
        i = 10
        return i + 15

    class C:

        def __lshift__(self, other):
            if False:
                return 10
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    c << 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaLshift()
except UnboundLocalError:
    print('OK, object lshift correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaRshift():
    if False:
        while True:
            i = 10

    class C:

        def __rshift__(self, other):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    c >> 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaRshift()
except UnboundLocalError:
    print('OK, object rshift correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaBitwiseAnd():
    if False:
        print('Hello World!')

    class C:

        def __and__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    c & 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaBitwiseAnd()
except UnboundLocalError:
    print('OK, object bitwise and correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaBitwiseOr():
    if False:
        while True:
            i = 10

    class C:

        def __or__(self, other):
            if False:
                print('Hello World!')
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    c | 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaBitwiseOr()
except UnboundLocalError:
    print('OK, object bitwise or correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaBitwiseXor():
    if False:
        i = 10
        return i + 15

    class C:

        def __xor__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    c ^ 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaBitwiseXor()
except UnboundLocalError:
    print('OK, object bitwise xor correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInt():
    if False:
        for i in range(10):
            print('nop')

    class C:

        def __int__(self):
            if False:
                print('Hello World!')
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    int(c)
    return a
try:
    someFunctionThatReturnsDeletedValueViaInt()
except UnboundLocalError:
    print('OK, object int correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaFloat():
    if False:
        i = 10
        return i + 15

    class C:

        def __float__(self):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
            return 0.0
    c = C()
    a = 1
    float(c)
    return a
try:
    someFunctionThatReturnsDeletedValueViaFloat()
except UnboundLocalError:
    print('OK, object float correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaComplex():
    if False:
        i = 10
        return i + 15

    class C:

        def __complex__(self):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
            return 0j
    c = C()
    a = 1
    complex(c)
    return a
try:
    someFunctionThatReturnsDeletedValueViaComplex()
except UnboundLocalError:
    print('OK, object complex correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInplaceAdd():
    if False:
        return 10

    class C:

        def __iadd__(self, other):
            if False:
                i = 10
                return i + 15
            nonlocal a
            del a
    c = C()
    a = 1
    c += 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaInplaceAdd()
except UnboundLocalError:
    print('OK, object inplace add correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInplaceSub():
    if False:
        while True:
            i = 10

    class C:

        def __isub__(self, other):
            if False:
                print('Hello World!')
            nonlocal a
            del a
    c = C()
    a = 1
    c -= 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaInplaceSub()
except UnboundLocalError:
    print('OK, object inplace sub correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInplaceMul():
    if False:
        i = 10
        return i + 15

    class C:

        def __imul__(self, other):
            if False:
                return 10
            nonlocal a
            del a
    c = C()
    a = 1
    c *= 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaInplaceMul()
except UnboundLocalError:
    print('OK, object inplace mul correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInplaceRemainder():
    if False:
        print('Hello World!')

    class C:

        def __imod__(self, other):
            if False:
                i = 10
                return i + 15
            nonlocal a
            del a
    c = C()
    a = 1
    c %= 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaInplaceRemainder()
except UnboundLocalError:
    print('OK, object inplace remainder correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInplacePower():
    if False:
        for i in range(10):
            print('nop')

    class C:

        def __ipow__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
            return 7
    c = C()
    a = 1
    c **= 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaInplacePower()
except UnboundLocalError:
    print('OK, object inplace power correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInplaceAnd():
    if False:
        i = 10
        return i + 15

    class C:

        def __iand__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    c &= 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaInplaceAnd()
except UnboundLocalError:
    print('OK, object inplace and correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInplaceFloordiv():
    if False:
        i = 10
        return i + 15

    class C:

        def __ifloordiv__(self, other):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
            return 7
    c = C()
    a = 1
    c //= 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaInplaceFloordiv()
except UnboundLocalError:
    print('OK, object inplace floordiv correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInplaceLshift():
    if False:
        print('Hello World!')

    class C:

        def __ilshift__(self, other):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    c <<= 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaInplaceLshift()
except UnboundLocalError:
    print('OK, object inplace lshift correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInplaceRshift():
    if False:
        return 10

    class C:

        def __irshift__(self, other):
            if False:
                i = 10
                return i + 15
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    c >>= 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaInplaceRshift()
except UnboundLocalError:
    print('OK, object inplace rshift correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInplaceOr():
    if False:
        i = 10
        return i + 15

    class C:

        def __ior__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    c |= 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaInplaceOr()
except UnboundLocalError:
    print('OK, object inplace or correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInplaceTrueDiv():
    if False:
        return 10

    class C:

        def __itruediv__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
            return 7
    c = C()
    a = 1
    c /= 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaInplaceTrueDiv()
except UnboundLocalError:
    print('OK, object inplace truediv correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInplaceXor():
    if False:
        return 10

    class C:

        def __ixor__(self, other):
            if False:
                return 10
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    c ^= 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaInplaceXor()
except UnboundLocalError:
    print('OK, object inplace xor correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaIndex():
    if False:
        return 10

    class C:

        def __index__(self):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
            return 0
    c = C()
    a = 1
    [1][c]
    return a
try:
    someFunctionThatReturnsDeletedValueViaIndex()
except UnboundLocalError:
    print('OK, object index correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaLen():
    if False:
        return 10

    class C:

        def __len__(self):
            if False:
                i = 10
                return i + 15
            nonlocal a
            del a
            return 0
    c = C()
    a = 1
    len(c)
    return a
try:
    someFunctionThatReturnsDeletedValueViaLen()
except UnboundLocalError:
    print('OK, object len correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaRepr():
    if False:
        while True:
            i = 10

    class C:

        def __repr__(self):
            if False:
                return 10
            nonlocal a
            del a
            return '<some_repr>'
    c = C()
    a = 1
    repr(c)
    return a
try:
    someFunctionThatReturnsDeletedValueViaRepr()
except UnboundLocalError:
    print('OK, object repr correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaStr():
    if False:
        return 10

    class C:

        def __str__(self):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
            return '<some_repr>'
    c = C()
    a = 1
    str(c)
    return a
try:
    someFunctionThatReturnsDeletedValueViaStr()
except UnboundLocalError:
    print('OK, object str correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaCompare():
    if False:
        print('Hello World!')

    class C:

        def __lt__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
            return '<some_repr>'
    c = C()
    a = 1
    c < None
    return a
try:
    someFunctionThatReturnsDeletedValueViaCompare()
except UnboundLocalError:
    print('OK, object compare correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaDel():
    if False:
        print('Hello World!')

    class C:

        def __del__(self):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
            return '<some_repr>'
    c = C()
    a = 1
    del c
    return a
try:
    someFunctionThatReturnsDeletedValueViaDel()
except UnboundLocalError:
    print('OK, object del correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaHash():
    if False:
        return 10

    class C:

        def __hash__(self):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
            return 42
    c = C()
    a = 1
    {}[c] = 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaHash()
except UnboundLocalError:
    print('OK, object hash correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaIter():
    if False:
        print('Hello World!')

    class C:

        def __iter__(self):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
            return iter(range(2))
    c = C()
    a = 1
    (x, y) = c
    return (a, x, y)
try:
    someFunctionThatReturnsDeletedValueViaIter()
except UnboundLocalError:
    print('OK, object iter correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaBytes():
    if False:
        for i in range(10):
            print('nop')

    class C:

        def __bytes__(self):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
            return bytes(range(2))
    c = C()
    a = 1
    bytes(c)
    return (a, x, y)
try:
    someFunctionThatReturnsDeletedValueViaBytes()
except UnboundLocalError:
    print('OK, object bytes correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaEq():
    if False:
        i = 10
        return i + 15

    class C:

        def __eq__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    c == 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaEq()
except UnboundLocalError:
    print('OK, object eq correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaLe():
    if False:
        i = 10
        return i + 15

    class C:

        def __le__(self, other):
            if False:
                return 10
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    c <= 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaEq()
except UnboundLocalError:
    print('OK, object le correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaGt():
    if False:
        i = 10
        return i + 15

    class C:

        def __gt__(self, other):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    c > 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaEq()
except UnboundLocalError:
    print('OK, object gt correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaGe():
    if False:
        i = 10
        return i + 15

    class C:

        def __ge__(self, other):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    c >= 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaEq()
except UnboundLocalError:
    print('OK, object ge correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaNe():
    if False:
        for i in range(10):
            print('nop')

    class C:

        def __ne__(self, other):
            if False:
                return 10
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    c != 1
    return a
try:
    someFunctionThatReturnsDeletedValueViaEq()
except UnboundLocalError:
    print('OK, object ne correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaContains():
    if False:
        for i in range(10):
            print('nop')

    class C:

        def __contains__(self, item):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal a
            del a
            return False
    c = C()
    a = 1
    1 in c
    return a
try:
    someFunctionThatReturnsDeletedValueViaContains()
except UnboundLocalError:
    print('OK, object contains correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaInit():
    if False:
        for i in range(10):
            print('nop')

    class C:

        def __init__(self):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
    a = 1
    c = C()
    return a
try:
    someFunctionThatReturnsDeletedValueViaInit()
except UnboundLocalError:
    print('OK, object init correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaNew():
    if False:
        for i in range(10):
            print('nop')

    class C:

        def __new__(self):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
    a = 1
    c = C()
    return a
try:
    someFunctionThatReturnsDeletedValueViaNew()
except UnboundLocalError:
    print('OK, object new correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaDir():
    if False:
        print('Hello World!')

    class C:

        def __dir__(self):
            if False:
                return 10
            nonlocal a
            del a
            return []
    c = C()
    a = 1
    dir(c)
    return a
try:
    someFunctionThatReturnsDeletedValueViaDir()
except UnboundLocalError:
    print('OK, object dir correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaReversed():
    if False:
        i = 10
        return i + 15

    class C:

        def __reversed__(self):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
            return None
    a = 1
    c = C()
    reversed(c)
    return a
try:
    someFunctionThatReturnsDeletedValueViaReversed()
except UnboundLocalError:
    print('OK, object reversed correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaFormat():
    if False:
        print('Hello World!')

    class C:

        def __format__(self, string):
            if False:
                return 10
            nonlocal a
            del a
            return 'formatted string'
    c = C()
    a = 1
    format(c, 'some string')
    return a
try:
    someFunctionThatReturnsDeletedValueViaFormat()
except UnboundLocalError:
    print('OK, object format correctly deleted an item.')
else:
    print('Ouch.!')

def someFunctionThatReturnsDeletedValueViaAbs():
    if False:
        return 10

    class C:

        def __abs__(self):
            if False:
                while True:
                    i = 10
            nonlocal a
            del a
            return abs(10)
    a = 1
    c = C()
    abs(c)
    return a
try:
    someFunctionThatReturnsDeletedValueViaAbs()
except UnboundLocalError:
    print('OK, object abs correctly deleted an item.')
else:
    print('Ouch.!')