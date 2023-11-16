from allennlp.data.fields import Field

def test_eq_with_inheritance():
    if False:
        while True:
            i = 10

    class SubField(Field):
        __slots__ = ['a']

        def __init__(self, a):
            if False:
                for i in range(10):
                    print('nop')
            self.a = a

    class SubSubField(SubField):
        __slots__ = ['b']

        def __init__(self, a, b):
            if False:
                i = 10
                return i + 15
            super().__init__(a)
            self.b = b

    class SubSubSubField(SubSubField):
        __slots__ = ['c']

        def __init__(self, a, b, c):
            if False:
                while True:
                    i = 10
            super().__init__(a, b)
            self.c = c
    assert SubField(1) == SubField(1)
    assert SubField(1) != SubField(2)
    assert SubSubField(1, 2) == SubSubField(1, 2)
    assert SubSubField(1, 2) != SubSubField(1, 1)
    assert SubSubField(1, 2) != SubSubField(2, 2)
    assert SubSubSubField(1, 2, 3) == SubSubSubField(1, 2, 3)
    assert SubSubSubField(1, 2, 3) != SubSubSubField(0, 2, 3)

def test_eq_with_inheritance_for_non_slots_field():
    if False:
        while True:
            i = 10

    class SubField(Field):

        def __init__(self, a):
            if False:
                while True:
                    i = 10
            self.a = a
    assert SubField(1) == SubField(1)
    assert SubField(1) != SubField(2)

def test_eq_with_inheritance_for_mixed_field():
    if False:
        print('Hello World!')

    class SubField(Field):
        __slots__ = ['a']

        def __init__(self, a):
            if False:
                while True:
                    i = 10
            self.a = a

    class SubSubField(SubField):

        def __init__(self, a, b):
            if False:
                print('Hello World!')
            super().__init__(a)
            self.b = b
    assert SubField(1) == SubField(1)
    assert SubField(1) != SubField(2)
    assert SubSubField(1, 2) == SubSubField(1, 2)
    assert SubSubField(1, 2) != SubSubField(1, 1)
    assert SubSubField(1, 2) != SubSubField(2, 2)