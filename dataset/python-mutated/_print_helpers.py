"""
Base class to provide str and repr hooks that `init_printing` can overwrite.

This is exposed publicly in the `printing.defaults` module,
but cannot be defined there without causing circular imports.
"""

class Printable:
    """
    The default implementation of printing for SymPy classes.

    This implements a hack that allows us to print elements of built-in
    Python containers in a readable way. Natively Python uses ``repr()``
    even if ``str()`` was explicitly requested. Mix in this trait into
    a class to get proper default printing.

    This also adds support for LaTeX printing in jupyter notebooks.
    """
    __slots__ = ()

    def __str__(self):
        if False:
            i = 10
            return i + 15
        from sympy.printing.str import sstr
        return sstr(self, order=None)
    __repr__ = __str__

    def _repr_disabled(self):
        if False:
            print('Hello World!')
        '\n        No-op repr function used to disable jupyter display hooks.\n\n        When :func:`sympy.init_printing` is used to disable certain display\n        formats, this function is copied into the appropriate ``_repr_*_``\n        attributes.\n\n        While we could just set the attributes to `None``, doing it this way\n        allows derived classes to call `super()`.\n        '
        return None
    _repr_png_ = _repr_disabled
    _repr_svg_ = _repr_disabled

    def _repr_latex_(self):
        if False:
            while True:
                i = 10
        '\n        IPython/Jupyter LaTeX printing\n\n        To change the behavior of this (e.g., pass in some settings to LaTeX),\n        use init_printing(). init_printing() will also enable LaTeX printing\n        for built in numeric types like ints and container types that contain\n        SymPy objects, like lists and dictionaries of expressions.\n        '
        from sympy.printing.latex import latex
        s = latex(self, mode='plain')
        return '$\\displaystyle %s$' % s