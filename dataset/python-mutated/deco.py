import inspect

def not_keyword(func):
    if False:
        return 10
    'Decorator to disable exposing functions or methods as keywords.\n\n    Examples::\n\n        @not_keyword\n        def not_exposed_as_keyword():\n            # ...\n\n        def exposed_as_keyword():\n            # ...\n\n    Alternatively the automatic keyword discovery can be disabled with\n    the :func:`library` decorator or by setting the ``ROBOT_AUTO_KEYWORDS``\n    attribute to a false value.\n\n    New in Robot Framework 3.2.\n    '
    func.robot_not_keyword = True
    return func
not_keyword.robot_not_keyword = True

@not_keyword
def keyword(name=None, tags=(), types=()):
    if False:
        i = 10
        return i + 15
    'Decorator to set custom name, tags and argument types to keywords.\n\n    This decorator creates ``robot_name``, ``robot_tags`` and ``robot_types``\n    attributes on the decorated keyword function or method based on the\n    provided arguments. Robot Framework checks them to determine the keyword\'s\n    name, tags, and argument types, respectively.\n\n    Name must be given as a string, tags as a list of strings, and types\n    either as a dictionary mapping argument names to types or as a list\n    of types mapped to arguments based on position. It is OK to specify types\n    only to some arguments, and setting ``types`` to ``None`` disables type\n    conversion altogether.\n\n    If the automatic keyword discovery has been disabled with the\n    :func:`library` decorator or by setting the ``ROBOT_AUTO_KEYWORDS``\n    attribute to a false value, this decorator is needed to mark functions\n    or methods keywords.\n\n    Examples::\n\n        @keyword\n        def example():\n            # ...\n\n        @keyword(\'Login as user "${user}" with password "${password}"\',\n                 tags=[\'custom name\', \'embedded arguments\', \'tags\'])\n        def login(user, password):\n            # ...\n\n        @keyword(types={\'length\': int, \'case_insensitive\': bool})\n        def types_as_dict(length, case_insensitive):\n            # ...\n\n        @keyword(types=[int, bool])\n        def types_as_list(length, case_insensitive):\n            # ...\n\n        @keyword(types=None])\n        def no_conversion(length, case_insensitive=False):\n            # ...\n    '
    if inspect.isroutine(name):
        return keyword()(name)

    def decorator(func):
        if False:
            for i in range(10):
                print('nop')
        func.robot_name = name
        func.robot_tags = tags
        func.robot_types = types
        return func
    return decorator

@not_keyword
def library(scope=None, version=None, converters=None, doc_format=None, listener=None, auto_keywords=False):
    if False:
        print('Hello World!')
    "Class decorator to control keyword discovery and other library settings.\n\n    By default disables automatic keyword detection by setting class attribute\n    ``ROBOT_AUTO_KEYWORDS = False`` to the decorated library. In that mode\n    only methods decorated explicitly with the :func:`keyword` decorator become\n    keywords. If that is not desired, automatic keyword discovery can be\n    enabled by using ``auto_keywords=True``.\n\n    Arguments ``scope``, ``version``, ``converters``, ``doc_format`` and ``listener``\n    set library's scope, version, converters, documentation format and listener by\n    using class attributes ``ROBOT_LIBRARY_SCOPE``, ``ROBOT_LIBRARY_VERSION``,\n    ``ROBOT_LIBRARY_CONVERTERS``, ``ROBOT_LIBRARY_DOC_FORMAT`` and\n    ``ROBOT_LIBRARY_LISTENER``, respectively. These attributes are only set if\n    the related arguments are given and they override possible existing attributes\n    in the decorated class.\n\n    Examples::\n\n        @library\n        class KeywordDiscovery:\n\n            @keyword\n            def do_something(self):\n                # ...\n\n            def not_keyword(self):\n                # ...\n\n\n        @library(scope='GLOBAL', version='3.2')\n        class LibraryConfiguration:\n            # ...\n\n    The ``@library`` decorator is new in Robot Framework 3.2.\n    The ``converters`` argument is new in Robot Framework 5.0.\n    "
    if inspect.isclass(scope):
        return library()(scope)

    def decorator(cls):
        if False:
            while True:
                i = 10
        if scope is not None:
            cls.ROBOT_LIBRARY_SCOPE = scope
        if version is not None:
            cls.ROBOT_LIBRARY_VERSION = version
        if converters is not None:
            cls.ROBOT_LIBRARY_CONVERTERS = converters
        if doc_format is not None:
            cls.ROBOT_LIBRARY_DOC_FORMAT = doc_format
        if listener is not None:
            cls.ROBOT_LIBRARY_LISTENER = listener
        cls.ROBOT_AUTO_KEYWORDS = auto_keywords
        return cls
    return decorator