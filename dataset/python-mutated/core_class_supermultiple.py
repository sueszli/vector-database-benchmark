"""
categories: Core,Classes
description: When inheriting from multiple classes super() only calls one class
cause: See :ref:`cpydiff_core_class_mro`
workaround: See :ref:`cpydiff_core_class_mro`
"""

class A:

    def __init__(self):
        if False:
            return 10
        print('A.__init__')

class B(A):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        print('B.__init__')
        super().__init__()

class C(A):

    def __init__(self):
        if False:
            return 10
        print('C.__init__')
        super().__init__()

class D(B, C):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        print('D.__init__')
        super().__init__()
D()