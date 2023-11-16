"""
categories: Core,Classes
description: Calling super() getter property in subclass will return a property object, not the value
cause: Unknown
workaround: Unknown
"""

class A:

    @property
    def p(self):
        if False:
            return 10
        return {'a': 10}

class AA(A):

    @property
    def p(self):
        if False:
            i = 10
            return i + 15
        return super().p
a = AA()
print(a.p)