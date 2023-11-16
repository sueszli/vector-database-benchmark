import inspect

def _partialmethod(method, *args1, **kwargs1):
    if False:
        print('Hello World!')

    def wrapper(self, *args2, **kwargs2):
        if False:
            print('Hello World!')
        return method(self, *args1, *args2, **kwargs1, **kwargs2)
    return wrapper

class Operation:
    """Descriptor used to define operations for delegating mixins.

    This class is designed to be assigned to the attributes (the delegating
    methods) defined by the OperationMixin. This class will create the method
    and mimic all the expected attributes for that method to appear as though
    it was originally designed on the class. The use of the descriptor pattern
    ensures that the method is only created the first time it is invoked, after
    which all further calls use the callable generated on the first invocation.

    Parameters
    ----------
    name : str
        The name of the operation.
    docstring_format_args : str
        The attribute of the owning class from which to pull format parameters
        for this operation's docstring.
    base_operation : str
        The underlying operation function to be invoked when operation `name`
        is called on the owning class.
    """

    def __init__(self, name, docstring_format_args, base_operation):
        if False:
            while True:
                i = 10
        self._name = name
        self._docstring_format_args = docstring_format_args
        self._base_operation = base_operation

    def __get__(self, obj, owner=None):
        if False:
            i = 10
            return i + 15
        retfunc = _partialmethod(self._base_operation, op=self._name)
        retfunc.__name__ = self._name
        retfunc.__qualname__ = '.'.join([owner.__name__, self._name])
        retfunc.__module__ = self._base_operation.__module__
        if self._base_operation.__doc__ is not None:
            retfunc.__doc__ = self._base_operation.__doc__.format(cls=owner.__name__, op=self._name, **self._docstring_format_args)
        retfunc.__annotations__ = self._base_operation.__annotations__.copy()
        retfunc.__annotations__.pop('op', None)
        retfunc_params = [v for (k, v) in inspect.signature(self._base_operation).parameters.items() if k != 'op']
        retfunc.__signature__ = inspect.Signature(retfunc_params)
        setattr(owner, self._name, retfunc)
        if obj is None:
            return getattr(owner, self._name)
        else:
            return getattr(obj, self._name)

def _should_define_operation(cls, operation, base_operation_name):
    if False:
        return 10
    if operation not in dir(cls):
        return True
    if base_operation_name not in cls.__dict__:
        return False
    for base_cls in cls.__mro__:
        if base_cls is object:
            return True
        if operation in base_cls.__dict__:
            return isinstance(base_cls.__dict__[operation], Operation)
    assert False, 'Operation attribute not found in hierarchy.'

def _create_delegating_mixin(mixin_name, docstring, category_name, base_operation_name, supported_operations):
    if False:
        return 10
    'Factory for mixins defining collections of delegated operations.\n\n    This function generates mixins based on two common paradigms in cuDF:\n\n    1. libcudf groups many operations into categories using a common API. These\n       APIs usually accept an enum to delineate the specific operation to\n       perform, e.g. binary operations use the `binary_operator` enum when\n       calling the `binary_operation` function. cuDF Python mimics this\n       structure by having operations within a category delegate to a common\n       internal function (e.g. DataFrame.__add__ calls DataFrame._binaryop).\n    2. Many cuDF classes implement similar operations (e.g. `sum`) via\n       delegation to lower-level APIs before reaching a libcudf C++ function\n       call. As a result, many API function calls actually involve multiple\n       delegations to lower-level APIs that can look essentially identical. An\n       example of such a sequence would be DataFrame.sum -> DataFrame._reduce\n       -> Column.sum -> Column._reduce -> libcudf.\n\n    This factory creates mixins for a category of operations implemented by via\n    this delegator pattern. The resulting mixins make it easy to share common\n    functions across various classes while also providing a common entrypoint\n    for implementing the centralized logic for a given category of operations.\n    Its usage is best demonstrated by example below.\n\n    Parameters\n    ----------\n    mixin_name : str\n        The name of the class. This argument should be the same as the object\n        that this function\'s output is assigned to, e.g.\n        :code:`Baz = _create_delegating_mixin("Baz", ...)`.\n    docstring : str\n        The documentation string for the mixin class.\n    category_name : str\n        The category of operations for which a mixin is being created. This\n        name will be used to define or access the following attributes as shown\n        in the example below:\n            - f\'_{category_name}_DOCSTRINGS\'\n            - f\'_VALID_{category_name}S\'  # The subset of ops a subclass allows\n            - f\'_SUPPORTED_{category_name}S\'  # The ops supported by the mixin\n    base_operation_name : str\n        The name given to the core function implementing this category of\n        operations.  The corresponding function is the entrypoint for child\n        classes.\n    supported_ops : List[str]\n        The list of valid operations that subclasses of the resulting mixin may\n        request to be implemented.\n\n    Examples\n    --------\n    >>> # The class below:\n    >>> class Person:\n    ...     def _greet(self, op):\n    ...         print(op)\n    ...\n    ...     def hello(self):\n    ...         self._greet("hello")\n    ...\n    ...     def goodbye(self):\n    ...         self._greet("goodbye")\n    >>> # can  be rewritten using a delegating mixin as follows:\n    >>> Greeter = _create_delegating_mixin(\n    ...     "Greeter", "", "GREETING", "_greet", {"hello", "goodbye", "hey"}\n    ... )\n    >>> # The `hello` and `goodbye` methods will now be automatically generated\n    >>> # for the Person class below.\n    >>> class Person(Greeter):\n    ...     _VALID_GREETINGS = {"hello", "goodbye"}\n    ...\n    ...     def _greet(self, op: str):\n    ...         \'\'\'Say {op}.\'\'\'\n    ...         print(op)\n    >>> mom = Person()\n    >>> mom.hello()\n    hello\n    >>> # The Greeter class could also enable the `hey` method, but Person did\n    >>> # not include it in the _VALID_GREETINGS set so it will not exist.\n    >>> mom.hey()\n    Traceback (most recent call last):\n        ...\n    AttributeError: \'Person\' object has no attribute \'hey\'\n    >>> # The docstrings for each method are generated by formatting the _greet\n    >>> # docstring with the operation name as well as any additional keys\n    >>> # provided via the _GREETING_DOCSTRINGS parameter.\n    >>> print(mom.hello.__doc__)\n    Say hello.\n    '
    validity_attr = f'_VALID_{category_name}S'
    docstring_attr = f'_{category_name}_DOCSTRINGS'
    supported_attr = f'_SUPPORTED_{category_name}S'

    class OperationMixin:

        @classmethod
        def __init_subclass__(cls):
            if False:
                print('Hello World!')
            super().__init_subclass__()
            valid_operations = set()
            for base_cls in cls.__mro__:
                valid_operations |= getattr(base_cls, validity_attr, set())
            invalid_operations = valid_operations - supported_operations
            assert len(invalid_operations) == 0, f'Invalid requested operations: {invalid_operations}'
            base_operation = getattr(cls, base_operation_name)
            for operation in valid_operations:
                if _should_define_operation(cls, operation, base_operation_name):
                    docstring_format_args = getattr(cls, docstring_attr, {}).get(operation, {})
                    op_attr = Operation(operation, docstring_format_args, base_operation)
                    setattr(cls, operation, op_attr)
    OperationMixin.__name__ = mixin_name
    OperationMixin.__qualname__ = mixin_name
    OperationMixin.__doc__ = docstring

    def _operation(self, op: str, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError
    _operation.__name__ = base_operation_name
    _operation.__qualname__ = '.'.join([mixin_name, base_operation_name])
    _operation.__doc__ = f'The core {category_name.lower()} function. Must be overridden by subclasses, the default implementation raises a NotImplementedError.'
    setattr(OperationMixin, base_operation_name, _operation)
    setattr(OperationMixin, supported_attr, supported_operations)
    return OperationMixin