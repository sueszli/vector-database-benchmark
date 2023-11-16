class ClassSimplest:
    pass

class ClassWithSingleField:
    a = 1

class ClassWithJustTheDocstring:
    """Just a docstring."""

class ClassWithInit:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class ClassWithTheDocstringAndInit:
    """Just a docstring."""

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

class ClassWithInitAndVars:
    cls_var = 100

    def __init__(self):
        if False:
            return 10
        pass

class ClassWithInitAndVarsAndDocstring:
    """Test class"""
    cls_var = 100

    def __init__(self):
        if False:
            return 10
        pass

class ClassWithDecoInit:

    @deco
    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

class ClassWithDecoInitAndVars:
    cls_var = 100

    @deco
    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

class ClassWithDecoInitAndVarsAndDocstring:
    """Test class"""
    cls_var = 100

    @deco
    def __init__(self):
        if False:
            while True:
                i = 10
        pass

class ClassSimplestWithInner:

    class Inner:
        pass

class ClassSimplestWithInnerWithDocstring:

    class Inner:
        """Just a docstring."""

        def __init__(self):
            if False:
                while True:
                    i = 10
            pass

class ClassWithSingleFieldWithInner:
    a = 1

    class Inner:
        pass

class ClassWithJustTheDocstringWithInner:
    """Just a docstring."""

    class Inner:
        pass

class ClassWithInitWithInner:

    class Inner:
        pass

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class ClassWithInitAndVarsWithInner:
    cls_var = 100

    class Inner:
        pass

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class ClassWithInitAndVarsAndDocstringWithInner:
    """Test class"""
    cls_var = 100

    class Inner:
        pass

    def __init__(self):
        if False:
            return 10
        pass

class ClassWithDecoInitWithInner:

    class Inner:
        pass

    @deco
    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class ClassWithDecoInitAndVarsWithInner:
    cls_var = 100

    class Inner:
        pass

    @deco
    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class ClassWithDecoInitAndVarsAndDocstringWithInner:
    """Test class"""
    cls_var = 100

    class Inner:
        pass

    @deco
    def __init__(self):
        if False:
            print('Hello World!')
        pass

class ClassWithDecoInitAndVarsAndDocstringWithInner2:
    """Test class"""

    class Inner:
        pass
    cls_var = 100

    @deco
    def __init__(self):
        if False:
            while True:
                i = 10
        pass

class ClassSimplest:
    pass

class ClassWithSingleField:
    a = 1

class ClassWithJustTheDocstring:
    """Just a docstring."""

class ClassWithInit:

    def __init__(self):
        if False:
            print('Hello World!')
        pass

class ClassWithTheDocstringAndInit:
    """Just a docstring."""

    def __init__(self):
        if False:
            print('Hello World!')
        pass

class ClassWithInitAndVars:
    cls_var = 100

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

class ClassWithInitAndVarsAndDocstring:
    """Test class"""
    cls_var = 100

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

class ClassWithDecoInit:

    @deco
    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

class ClassWithDecoInitAndVars:
    cls_var = 100

    @deco
    def __init__(self):
        if False:
            while True:
                i = 10
        pass

class ClassWithDecoInitAndVarsAndDocstring:
    """Test class"""
    cls_var = 100

    @deco
    def __init__(self):
        if False:
            return 10
        pass

class ClassSimplestWithInner:

    class Inner:
        pass

class ClassSimplestWithInnerWithDocstring:

    class Inner:
        """Just a docstring."""

        def __init__(self):
            if False:
                return 10
            pass

class ClassWithSingleFieldWithInner:
    a = 1

    class Inner:
        pass

class ClassWithJustTheDocstringWithInner:
    """Just a docstring."""

    class Inner:
        pass

class ClassWithInitWithInner:

    class Inner:
        pass

    def __init__(self):
        if False:
            return 10
        pass

class ClassWithInitAndVarsWithInner:
    cls_var = 100

    class Inner:
        pass

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

class ClassWithInitAndVarsAndDocstringWithInner:
    """Test class"""
    cls_var = 100

    class Inner:
        pass

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class ClassWithDecoInitWithInner:

    class Inner:
        pass

    @deco
    def __init__(self):
        if False:
            while True:
                i = 10
        pass

class ClassWithDecoInitAndVarsWithInner:
    cls_var = 100

    class Inner:
        pass

    @deco
    def __init__(self):
        if False:
            return 10
        pass

class ClassWithDecoInitAndVarsAndDocstringWithInner:
    """Test class"""
    cls_var = 100

    class Inner:
        pass

    @deco
    def __init__(self):
        if False:
            print('Hello World!')
        pass

class ClassWithDecoInitAndVarsAndDocstringWithInner2:
    """Test class"""

    class Inner:
        pass
    cls_var = 100

    @deco
    def __init__(self):
        if False:
            print('Hello World!')
        pass