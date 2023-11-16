"""This is the docstring for the example.py module.  Modules names should
have short, all-lowercase names.  The module name may have underscores if
this improves readability.

Every module should have a docstring at the very top of the file.  The
module's docstring may extend over multiple lines.  If your docstring does
extend over multiple lines, the closing three quotation marks must be on
a line by itself, preferably preceded by a blank line.

"""
import os
from .expected import Expectation
expectation = Expectation()
expect = expectation.expect
expectation.expected.add((os.path.normcase(__file__), 'D205: 1 blank line required between summary line and description (found 0)'))
expectation.expected.add((os.path.normcase(__file__), 'D213: Multi-line docstring summary should start at the second line'))
expectation.expected.add((os.path.normcase(__file__), "D400: First line should end with a period (not 'd')"))
expectation.expected.add((os.path.normcase(__file__), 'D404: First word of the docstring should not be `This`'))
expectation.expected.add((os.path.normcase(__file__), "D415: First line should end with a period, question mark, or exclamation point (not 'd')"))

@expect('D213: Multi-line docstring summary should start at the second line', arg_count=3)
@expect("D401: First line should be in imperative mood; try rephrasing (found 'A')", arg_count=3)
@expect("D413: Missing blank line after last section ('Examples')", arg_count=3)
def foo(var1, var2, long_var_name='hi', **kwargs):
    if False:
        print('Hello World!')
    'A one-line summary that does not use variable names.\n\n    Several sentences providing an extended description. Refer to\n    variables using back-ticks, e.g. `var`.\n\n    Parameters\n    ----------\n    var1 : array_like\n        Array_like means all those objects -- lists, nested lists, etc. --\n        that can be converted to an array.  We can also refer to\n        variables like `var1`.\n    var2 : int\n        The type above can either refer to an actual Python type\n        (e.g. ``int``), or describe the type of the variable in more\n        detail, e.g. ``(N,) ndarray`` or ``array_like``.\n    long_var_name : {\'hi\', \'ho\'}, optional\n        Choices in brackets, default first when optional.\n    **kwargs : int\n        More keyword arguments.\n\n    Returns\n    -------\n    type\n        Explanation of anonymous return value of type ``type``.\n    describe : type\n        Explanation of return value named `describe`.\n    out : type\n        Explanation of `out`.\n    type_without_description\n\n    Other Parameters\n    ----------------\n    only_seldom_used_keywords : type\n        Explanation\n    common_parameters_listed_above : type\n        Explanation\n\n    Raises\n    ------\n    BadException\n        Because you shouldn\'t have done that.\n\n    See Also\n    --------\n    numpy.array : Relationship (optional).\n    numpy.ndarray : Relationship (optional), which could be fairly long, in\n                    which case the line wraps here.\n    numpy.dot, numpy.linalg.norm, numpy.eye\n\n    Notes\n    -----\n    Notes about the implementation algorithm (if needed).\n\n    This can have multiple paragraphs.\n\n    You may include some math:\n\n    .. math:: X(e^{j\\omega } ) = x(n)e^{ - j\\omega n}\n\n    And even use a Greek symbol like :math:`\\omega` inline.\n\n    References\n    ----------\n    Cite the relevant literature, e.g. [1]_.  You may also cite these\n    references in the notes section above.\n\n    .. [1] O. McNoleg, "The integration of GIS, remote sensing,\n       expert systems and adaptive co-kriging for environmental habitat\n       modelling of the Highland Haggis using object-oriented, fuzzy-logic\n       and neural-network techniques," Computers & Geosciences, vol. 22,\n       pp. 585-588, 1996.\n\n    Examples\n    --------\n    These are written in doctest format, and should illustrate how to\n    use the function.\n\n    >>> a = [1, 2, 3]\n    >>> print([x + 3 for x in a])\n    [4, 5, 6]\n    >>> print("a\\nb")\n    a\n    b\n    '
    pass