from .cartan_type import CartanType

def DynkinDiagram(t):
    if False:
        while True:
            i = 10
    'Display the Dynkin diagram of a given Lie algebra\n\n    Works by generating the CartanType for the input, t, and then returning the\n    Dynkin diagram method from the individual classes.\n\n    Examples\n    ========\n\n    >>> from sympy.liealgebras.dynkin_diagram import DynkinDiagram\n    >>> print(DynkinDiagram("A3"))\n    0---0---0\n    1   2   3\n\n    >>> print(DynkinDiagram("B4"))\n    0---0---0=>=0\n    1   2   3   4\n\n    '
    return CartanType(t).dynkin_diagram()