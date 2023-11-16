"""Example NumPy style docstrings.

This module demonstrates documentation as specified by the `NumPy
Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
are created with a section header followed by an underline of equal length.

Example
-------
Examples can be given using either the ``Example`` or ``Examples``
sections. Sections support any reStructuredText formatting, including
literal blocks::

    $ python example_numpy.py


Section breaks are created with two blank lines. Section breaks are also
implicitly created anytime a new section starts. Section bodies *may* be
indented:

Notes
-----
    This is an example of an indented section. It's like any other section,
    but the body is indented to help it stand out from surrounding text.

If a section is indented, then a section break is created by
resuming unindented text.

Attributes
----------
module_level_variable1 : int
    Module level variables may be documented in either the ``Attributes``
    section of the module docstring, or in an inline docstring immediately
    following the variable.

    Either form is acceptable, but the two should not be mixed. Choose
    one convention to document module level variables and be consistent
    with it.


.. _NumPy docstring standard:
   https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard

"""
module_level_variable1 = 12345
module_level_variable2 = 98765
'int: Module level variable documented inline.\n\nThe docstring may span multiple lines. The type may optionally be specified\non the first line, separated by a colon.\n'

def function_with_types_in_docstring(param1, param2):
    if False:
        return 10
    'Example function with types documented in the docstring.\n\n    :pep:`484` type annotations are supported. If attribute, parameter, and\n    return types are annotated according to `PEP 484`_, they do not need to be\n    included in the docstring:\n\n    Parameters\n    ----------\n    param1 : int\n        The first parameter.\n    param2 : str\n        The second parameter.\n\n    Returns\n    -------\n    bool\n        True if successful, False otherwise.\n    '

def function_with_pep484_type_annotations(param1: int, param2: str) -> bool:
    if False:
        return 10
    'Example function with PEP 484 type annotations.\n\n    The return type must be duplicated in the docstring to comply\n    with the NumPy docstring style.\n\n    Parameters\n    ----------\n    param1\n        The first parameter.\n    param2\n        The second parameter.\n\n    Returns\n    -------\n    bool\n        True if successful, False otherwise.\n\n    '

def module_level_function(param1, param2=None, *args, **kwargs):
    if False:
        return 10
    'This is an example of a module level function.\n\n    Function parameters should be documented in the ``Parameters`` section.\n    The name of each parameter is required. The type and description of each\n    parameter is optional, but should be included if not obvious.\n\n    If ``*args`` or ``**kwargs`` are accepted,\n    they should be listed as ``*args`` and ``**kwargs``.\n\n    The format for a parameter is::\n\n        name : type\n            description\n\n            The description may span multiple lines. Following lines\n            should be indented to match the first line of the description.\n            The ": type" is optional.\n\n            Multiple paragraphs are supported in parameter\n            descriptions.\n\n    Parameters\n    ----------\n    param1 : int\n        The first parameter.\n    param2 : :obj:`str`, optional\n        The second parameter.\n    *args\n        Variable length argument list.\n    **kwargs\n        Arbitrary keyword arguments.\n\n    Returns\n    -------\n    bool\n        True if successful, False otherwise.\n\n        The return type is not optional. The ``Returns`` section may span\n        multiple lines and paragraphs. Following lines should be indented to\n        match the first line of the description.\n\n        The ``Returns`` section supports any reStructuredText formatting,\n        including literal blocks::\n\n            {\n                \'param1\': param1,\n                \'param2\': param2\n            }\n\n    Raises\n    ------\n    AttributeError\n        The ``Raises`` section is a list of all exceptions\n        that are relevant to the interface.\n    ValueError\n        If `param2` is equal to `param1`.\n\n    '
    if param1 == param2:
        raise ValueError('param1 may not be equal to param2')
    return True

def example_generator(n):
    if False:
        print('Hello World!')
    'Generators have a ``Yields`` section instead of a ``Returns`` section.\n\n    Parameters\n    ----------\n    n : int\n        The upper limit of the range to generate, from 0 to `n` - 1.\n\n    Yields\n    ------\n    int\n        The next number in the range of 0 to `n` - 1.\n\n    Examples\n    --------\n    Examples should be written in doctest format, and should illustrate how\n    to use the function.\n\n    >>> print([i for i in example_generator(4)])\n    [0, 1, 2, 3]\n\n    '
    yield from range(n)

class ExampleError(Exception):
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Note
    ----
    Do not include the `self` parameter in the ``Parameters`` section.

    Parameters
    ----------
    msg : str
        Human readable string describing the exception.
    code : :obj:`int`, optional
        Numeric error code.

    Attributes
    ----------
    msg : str
        Human readable string describing the exception.
    code : int
        Numeric error code.

    """

    def __init__(self, msg, code):
        if False:
            while True:
                i = 10
        self.msg = msg
        self.code = code

class ExampleClass:
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes
    ----------
    attr1 : str
        Description of `attr1`.
    attr2 : :obj:`int`, optional
        Description of `attr2`.

    """

    def __init__(self, param1, param2, param3):
        if False:
            i = 10
            return i + 15
        'Example of docstring on the __init__ method.\n\n        The __init__ method may be documented in either the class level\n        docstring, or as a docstring on the __init__ method itself.\n\n        Either form is acceptable, but the two should not be mixed. Choose one\n        convention to document the __init__ method and be consistent with it.\n\n        Note\n        ----\n        Do not include the `self` parameter in the ``Parameters`` section.\n\n        Parameters\n        ----------\n        param1 : str\n            Description of `param1`.\n        param2 : list(str)\n            Description of `param2`. Multiple\n            lines are supported.\n        param3 : :obj:`int`, optional\n            Description of `param3`.\n\n        '
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = param3
        self.attr4 = ['attr4']
        self.attr5 = None
        'str: Docstring *after* attribute, with type specified.'

    @property
    def readonly_property(self):
        if False:
            while True:
                i = 10
        'str: Properties should be documented in their getter method.'
        return 'readonly_property'

    @property
    def readwrite_property(self):
        if False:
            print('Hello World!')
        'list(str): Properties with both a getter and setter\n        should only be documented in their getter method.\n\n        If the setter method contains notable behavior, it should be\n        mentioned here.\n        '
        return ['readwrite_property']

    @readwrite_property.setter
    def readwrite_property(self, value):
        if False:
            while True:
                i = 10
        value

    def example_method(self, param1, param2):
        if False:
            for i in range(10):
                print('nop')
        'Class methods are similar to regular functions.\n\n        Note\n        ----\n        Do not include the `self` parameter in the ``Parameters`` section.\n\n        Parameters\n        ----------\n        param1\n            The first parameter.\n        param2\n            The second parameter.\n\n        Returns\n        -------\n        bool\n            True if successful, False otherwise.\n\n        '
        return True

    def __special__(self):
        if False:
            print('Hello World!')
        "By default special members with docstrings are not included.\n\n        Special members are any methods or attributes that start with and\n        end with a double underscore. Any special member with a docstring\n        will be included in the output, if\n        ``napoleon_include_special_with_doc`` is set to True.\n\n        This behavior can be enabled by changing the following setting in\n        Sphinx's conf.py::\n\n            napoleon_include_special_with_doc = True\n\n        "
        pass

    def __special_without_docstring__(self):
        if False:
            print('Hello World!')
        pass

    def _private(self):
        if False:
            for i in range(10):
                print('nop')
        "By default private members are not included.\n\n        Private members are any methods or attributes that start with an\n        underscore and are *not* special. By default they are not included\n        in the output.\n\n        This behavior can be changed such that private members *are* included\n        by changing the following setting in Sphinx's conf.py::\n\n            napoleon_include_private_with_doc = True\n\n        "
        pass

    def _private_without_docstring(self):
        if False:
            while True:
                i = 10
        pass