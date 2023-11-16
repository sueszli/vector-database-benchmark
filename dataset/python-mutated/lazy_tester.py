"""Lazy testers for optional features."""
import abc
import contextlib
import functools
import importlib
import subprocess
import typing
from typing import Union, Iterable, Dict, Optional, Callable, Type
from qiskit.exceptions import MissingOptionalLibraryError
from .classtools import wrap_method

class _RequireNow:
    """Helper callable that accepts all function signatures and simply calls
    :meth:`.LazyDependencyManager.require_now`.  This helpful when used with :func:`.wrap_method`,
    as the callable needs to be compatible with all signatures and be picklable."""
    __slots__ = ('_tester', '_feature')

    def __init__(self, tester, feature):
        if False:
            return 10
        self._tester = tester
        self._feature = feature

    def __call__(self, *_args, **_kwargs):
        if False:
            return 10
        self._tester.require_now(self._feature)

class LazyDependencyManager(abc.ABC):
    """A mananger for some optional features that are expensive to import, or to verify the
    existence of.

    These objects can be used as Booleans, such as ``if x``, and will evaluate ``True`` if the
    dependency they test for is available, and ``False`` if not.  The presence of the dependency
    will only be tested when the Boolean is evaluated, so it can be used as a runtime test in
    functions and methods without requiring an import-time test.

    These objects also encapsulate the error handling if their dependency is not present, so you can
    do things such as::

        from qiskit.utils import LazyImportManager
        HAS_MATPLOTLIB = LazyImportManager("matplotlib")

        @HAS_MATPLOTLIB.require_in_call
        def my_visualisation():
            ...

        def my_other_visualisation():
            # ... some setup ...
            HAS_MATPLOTLIB.require_now("my_other_visualisation")
            ...

        def my_third_visualisation():
            if HAS_MATPLOTLIB:
                from matplotlib import pyplot
            else:
                ...

    In all of these cases, ``matplotlib`` is not imported until the functions are entered.  In the
    case of the decorator, ``matplotlib`` is tested for import when the function is called for
    the first time.  In the second and third cases, the loader attempts to import ``matplotlib``
    when the :meth:`require_now` method is called, or when the Boolean context is evaluated.  For
    the ``require`` methods, an error is raised if the library is not available.

    This is the base class, which provides the Boolean context checking and error management.  The
    concrete classes :class:`LazyImportTester` and :class:`LazySubprocessTester` provide convenient
    entry points for testing that certain symbols are importable from modules, or certain
    command-line tools are available, respectively.
    """
    __slots__ = ('_bool', '_callback', '_name', '_install', '_msg')

    def __init__(self, *, name=None, callback=None, install=None, msg=None):
        if False:
            while True:
                i = 10
        '\n        Args:\n            name: the name of this optional dependency.\n            callback: a callback that is called immediately after the availability of the library is\n                tested with the result.  This will only be called once.\n            install: how to install this optional dependency.  Passed to\n                :class:`.MissingOptionalLibraryError` as the ``pip_install`` parameter.\n            msg: an extra message to include in the error raised if this is required.\n        '
        self._bool = None
        self._callback = callback
        self._name = name
        self._install = install
        self._msg = msg

    @abc.abstractmethod
    def _is_available(self) -> bool:
        if False:
            print('Hello World!')
        'Subclasses of :class:`LazyDependencyManager` should override this method to implement the\n        actual test of availability.  This method should return a Boolean, where ``True`` indicates\n        that the dependency was available.  This method will only ever be called once.\n\n        :meta public:\n        '
        return False

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        if self._bool is None:
            self._bool = self._is_available()
            if self._callback is not None:
                self._callback(self._bool)
        return self._bool

    @typing.overload
    def require_in_call(self, feature_or_callable: Callable) -> Callable:
        if False:
            while True:
                i = 10
        ...

    @typing.overload
    def require_in_call(self, feature_or_callable: str) -> Callable[[Callable], Callable]:
        if False:
            return 10
        ...

    def require_in_call(self, feature_or_callable):
        if False:
            while True:
                i = 10
        'Create a decorator for callables that requires that the dependency is available when the\n        decorated function or method is called.\n\n        Args:\n            feature_or_callable (str or Callable): the name of the feature that requires these\n                dependencies.  If this function is called directly as a decorator (for example\n                ``@HAS_X.require_in_call`` as opposed to\n                ``@HAS_X.require_in_call("my feature")``), then the feature name will be taken to be\n                the function name, or class and method name as appropriate.\n\n        Returns:\n            Callable: a decorator that will make its argument require this dependency before it is\n            called.\n        '
        if isinstance(feature_or_callable, str):
            feature = feature_or_callable

            def decorator(function):
                if False:
                    return 10

                @functools.wraps(function)
                def out(*args, **kwargs):
                    if False:
                        print('Hello World!')
                    self.require_now(feature)
                    return function(*args, **kwargs)
                return out
            return decorator
        function = feature_or_callable
        feature = getattr(function, '__qualname__', None) or getattr(function, '__name__', None) or str(function)

        @functools.wraps(function)
        def out(*args, **kwargs):
            if False:
                print('Hello World!')
            self.require_now(feature)
            return function(*args, **kwargs)
        return out

    @typing.overload
    def require_in_instance(self, feature_or_class: Type) -> Type:
        if False:
            for i in range(10):
                print('nop')
        ...

    @typing.overload
    def require_in_instance(self, feature_or_class: str) -> Callable[[Type], Type]:
        if False:
            for i in range(10):
                print('nop')
        ...

    def require_in_instance(self, feature_or_class):
        if False:
            while True:
                i = 10
        'A class decorator that requires the dependency is available when the class is\n        initialised.  This decorator can be used even if the class does not define an ``__init__``\n        method.\n\n        Args:\n            feature_or_class (str or Type): the name of the feature that requires these\n                dependencies.  If this function is called directly as a decorator (for example\n                ``@HAS_X.require_in_instance`` as opposed to\n                ``@HAS_X.require_in_instance("my feature")``), then the feature name will be taken\n                as the name of the class.\n\n        Returns:\n            Callable: a class decorator that ensures that the wrapped feature is present if the\n            class is initialised.\n        '
        if isinstance(feature_or_class, str):
            feature = feature_or_class

            def decorator(class_):
                if False:
                    print('Hello World!')
                wrap_method(class_, '__init__', before=_RequireNow(self, feature))
                return class_
            return decorator
        class_ = feature_or_class
        feature = getattr(class_, '__qualname__', None) or getattr(class_, '__name__', None) or str(class_)
        wrap_method(class_, '__init__', before=_RequireNow(self, feature))
        return class_

    def require_now(self, feature: str):
        if False:
            return 10
        'Eagerly attempt to import the dependencies in this object, and raise an exception if they\n        cannot be imported.\n\n        Args:\n            feature: the name of the feature that is requiring these dependencies.\n\n        Raises:\n            MissingOptionalLibraryError: if the dependencies cannot be imported.\n        '
        if self:
            return
        raise MissingOptionalLibraryError(libname=self._name, name=feature, pip_install=self._install, msg=self._msg)

    @contextlib.contextmanager
    def disable_locally(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a context, during which the value of the dependency manager will be ``False``.  This\n        means that within the context, any calls to this object will behave as if the dependency is\n        not available, including raising errors.  It is valid to call this method whether or not the\n        dependency has already been evaluated.  This is most useful in tests.\n        '
        previous = self._bool
        self._bool = False
        try:
            yield
        finally:
            self._bool = previous

class LazyImportTester(LazyDependencyManager):
    """A lazy dependency tester for importable Python modules.  Any required objects will only be
    imported at the point that this object is tested for its Boolean value."""
    __slots__ = ('_modules',)

    def __init__(self, name_map_or_modules: Union[str, Dict[str, Iterable[str]], Iterable[str]], *, name: Optional[str]=None, callback: Optional[Callable[[bool], None]]=None, install: Optional[str]=None, msg: Optional[str]=None):
        if False:
            while True:
                i = 10
        '\n        Args:\n            name_map_or_modules: if a name map, then a dictionary where the keys are modules or\n                packages, and the values are iterables of names to try and import from that\n                module.  It should be valid to write ``from <module> import <name1>, <name2>, ...``.\n                If simply a string or iterable of strings, then it should be valid to write\n                ``import <module>`` for each of them.\n\n        Raises:\n            ValueError: if no modules are given.\n        '
        if isinstance(name_map_or_modules, dict):
            self._modules = {module: tuple(names) for (module, names) in name_map_or_modules.items()}
        elif isinstance(name_map_or_modules, str):
            self._modules = {name_map_or_modules: ()}
        else:
            self._modules = {module: () for module in name_map_or_modules}
        if not self._modules:
            raise ValueError('no modules supplied')
        if name is not None:
            pass
        elif len(self._modules) == 1:
            (name,) = self._modules.keys()
        else:
            all_names = tuple(self._modules.keys())
            name = f"{', '.join(all_names[:-1])} and {all_names[-1]}"
        super().__init__(name=name, callback=callback, install=install, msg=msg)

    def _is_available(self):
        if False:
            print('Hello World!')
        try:
            for (module, names) in self._modules.items():
                imported = importlib.import_module(module)
                for name in names:
                    getattr(imported, name)
        except (ImportError, AttributeError):
            return False
        return True

class LazySubprocessTester(LazyDependencyManager):
    """A lazy checker that a command-line tool is available.  The command will only be run once, at
    the point that this object is checked for its Boolean value.
    """
    __slots__ = ('_command',)

    def __init__(self, command: Union[str, Iterable[str]], *, name: Optional[str]=None, callback: Optional[Callable[[bool], None]]=None, install: Optional[str]=None, msg: Optional[str]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            command: the strings that make up the command to be run.  For example,\n                ``["pdflatex", "-version"]``.\n\n        Raises:\n            ValueError: if an empty command is given.\n        '
        self._command = (command,) if isinstance(command, str) else tuple(command)
        if not self._command:
            raise ValueError('no command supplied')
        super().__init__(name=name or self._command[0], callback=callback, install=install, msg=msg)

    def _is_available(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            subprocess.run(self._command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            return False
        else:
            return True