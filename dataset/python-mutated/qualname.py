from __future__ import print_function
import cython

def test_qualname():
    if False:
        for i in range(10):
            print('nop')
    "\n    >>> test_qualname.__qualname__\n    'test_qualname'\n    >>> test_qualname.__qualname__ = 123 #doctest:+ELLIPSIS\n    Traceback (most recent call last):\n    TypeError: __qualname__ must be set to a ... object\n    >>> test_qualname.__qualname__ = 'foo'\n    >>> test_qualname.__qualname__\n    'foo'\n    "

def test_builtin_qualname():
    if False:
        for i in range(10):
            print('nop')
    '\n    >>> test_builtin_qualname()\n    list.append\n    len\n    '
    print([1, 2, 3].append.__qualname__)
    print(len.__qualname__)

def test_nested_qualname():
    if False:
        print('Hello World!')
    "\n    >>> outer, lambda_func, XYZ = test_nested_qualname()\n    defining class XYZ XYZ qualname\n    defining class Inner XYZ.Inner qualname\n\n    >>> outer_result = outer()\n    defining class Test test_nested_qualname.<locals>.outer.<locals>.Test qualname\n    >>> outer_result.__qualname__\n    'test_nested_qualname.<locals>.outer.<locals>.Test'\n    >>> outer_result.test.__qualname__\n    'test_nested_qualname.<locals>.outer.<locals>.Test.test'\n\n    >>> outer_result().test.__qualname__\n    'test_nested_qualname.<locals>.outer.<locals>.Test.test'\n\n    >>> outer_result_test_result = outer_result().test()\n    defining class XYZInner XYZinner qualname\n    >>> outer_result_test_result.__qualname__\n    'XYZinner'\n    >>> outer_result_test_result.Inner.__qualname__\n    'XYZinner.Inner'\n    >>> outer_result_test_result.Inner.inner.__qualname__\n    'XYZinner.Inner.inner'\n\n    >>> lambda_func.__qualname__\n    'test_nested_qualname.<locals>.<lambda>'\n\n    >>> XYZ.__qualname__\n    'XYZ'\n    >>> XYZ.Inner.__qualname__\n    'XYZ.Inner'\n    >>> XYZ.Inner.inner.__qualname__\n    'XYZ.Inner.inner'\n    "

    def outer():
        if False:
            return 10

        class Test(object):
            print('defining class Test', __qualname__, __module__)

            def test(self):
                if False:
                    while True:
                        i = 10
                global XYZinner

                class XYZinner:
                    print('defining class XYZInner', __qualname__, __module__)

                    class Inner:

                        def inner(self):
                            if False:
                                while True:
                                    i = 10
                            pass
                return XYZinner
        return Test
    global XYZ

    class XYZ(object):
        print('defining class XYZ', __qualname__, __module__)

        class Inner(object):
            print('defining class Inner', __qualname__, __module__)

            def inner(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
    return (outer, lambda : None, XYZ)

@cython.cclass
class CdefClass:
    """
    >>> print(CdefClass.qn, CdefClass.m)
    CdefClass qualname
    >>> print(CdefClass.__qualname__, CdefClass.__module__)
    CdefClass qualname

    #>>> print(CdefClass.l["__qualname__"], CdefClass.l["__module__"])
    #CdefClass qualname
    """
    qn = __qualname__
    m = __module__

@cython.cclass
class CdefModifyNames:
    """
    >>> print(CdefModifyNames.qn_reassigned, CdefModifyNames.m_reassigned)
    I'm not a qualname I'm not a module

    # TODO - enable when https://github.com/cython/cython/issues/4815 is fixed
    #>>> hasattr(CdefModifyNames, "qn_deleted")
    #False
    #>>> hasattr(CdefModifyNames, "m_deleted")
    #False

    #>>> print(CdefModifyNames.l["__qualname__"], CdefModifyNames.l["__module__"])
    #I'm not a qualname I'm not a module
    """
    __qualname__ = "I'm not a qualname"
    __module__ = "I'm not a module"
    qn_reassigned = __qualname__
    m_reassigned = __module__