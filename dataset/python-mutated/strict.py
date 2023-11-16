from __future__ import annotations

def f():
    if False:
        return 10
    import pkg
    import pkg.bar

    def test(value: pkg.bar.A):
        if False:
            while True:
                i = 10
        return pkg.B()

def f():
    if False:
        for i in range(10):
            print('nop')
    import pkg
    import pkg.bar

    def test(value: pkg.A):
        if False:
            print('Hello World!')
        return pkg.bar.B()

def f():
    if False:
        while True:
            i = 10
    import pkg
    from pkg import A

    def test(value: A):
        if False:
            i = 10
            return i + 15
        return pkg.B()

def f():
    if False:
        while True:
            i = 10
    from pkg import A, B

    def test(value: A):
        if False:
            for i in range(10):
                print('nop')
        return B()

def f():
    if False:
        for i in range(10):
            print('nop')
    import pkg.bar
    import pkg.baz

    def test(value: pkg.bar.A):
        if False:
            return 10
        return pkg.baz.B()

def f():
    if False:
        print('Hello World!')
    import pkg
    from pkg.bar import A

    def test(value: A):
        if False:
            return 10
        return pkg.B()

def f():
    if False:
        print('Hello World!')
    import pkg
    import pkg.bar as B

    def test(value: pkg.A):
        if False:
            for i in range(10):
                print('nop')
        return B()

def f():
    if False:
        i = 10
        return i + 15
    import pkg.foo as F
    import pkg.foo.bar as B

    def test(value: F.Foo):
        if False:
            print('Hello World!')
        return B()

def f():
    if False:
        return 10
    import pkg
    import pkg.foo.bar as B

    def test(value: pkg.A):
        if False:
            i = 10
            return i + 15
        return B()

def f():
    if False:
        print('Hello World!')
    import pkg
    import pkgfoo.bar as B

    def test(value: pkg.A):
        if False:
            print('Hello World!')
        return B()

def f():
    if False:
        i = 10
        return i + 15
    import pkg.bar as B
    import pkg.foo as F

    def test(value: F.Foo):
        if False:
            while True:
                i = 10
        return B.Bar()