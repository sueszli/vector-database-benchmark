"""Monkeypatching and mocking functionality."""
import os
import re
import sys
import warnings
from contextlib import contextmanager
from typing import Any
from typing import final
from typing import Generator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import TypeVar
from typing import Union
from _pytest.fixtures import fixture
from _pytest.warning_types import PytestWarning
RE_IMPORT_ERROR_NAME = re.compile('^No module named (.*)$')
K = TypeVar('K')
V = TypeVar('V')

@fixture
def monkeypatch() -> Generator['MonkeyPatch', None, None]:
    if False:
        print('Hello World!')
    'A convenient fixture for monkey-patching.\n\n    The fixture provides these methods to modify objects, dictionaries, or\n    :data:`os.environ`:\n\n    * :meth:`monkeypatch.setattr(obj, name, value, raising=True) <pytest.MonkeyPatch.setattr>`\n    * :meth:`monkeypatch.delattr(obj, name, raising=True) <pytest.MonkeyPatch.delattr>`\n    * :meth:`monkeypatch.setitem(mapping, name, value) <pytest.MonkeyPatch.setitem>`\n    * :meth:`monkeypatch.delitem(obj, name, raising=True) <pytest.MonkeyPatch.delitem>`\n    * :meth:`monkeypatch.setenv(name, value, prepend=None) <pytest.MonkeyPatch.setenv>`\n    * :meth:`monkeypatch.delenv(name, raising=True) <pytest.MonkeyPatch.delenv>`\n    * :meth:`monkeypatch.syspath_prepend(path) <pytest.MonkeyPatch.syspath_prepend>`\n    * :meth:`monkeypatch.chdir(path) <pytest.MonkeyPatch.chdir>`\n    * :meth:`monkeypatch.context() <pytest.MonkeyPatch.context>`\n\n    All modifications will be undone after the requesting test function or\n    fixture has finished. The ``raising`` parameter determines if a :class:`KeyError`\n    or :class:`AttributeError` will be raised if the set/deletion operation does not have the\n    specified target.\n\n    To undo modifications done by the fixture in a contained scope,\n    use :meth:`context() <pytest.MonkeyPatch.context>`.\n    '
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()

def resolve(name: str) -> object:
    if False:
        i = 10
        return i + 15
    parts = name.split('.')
    used = parts.pop(0)
    found: object = __import__(used)
    for part in parts:
        used += '.' + part
        try:
            found = getattr(found, part)
        except AttributeError:
            pass
        else:
            continue
        try:
            __import__(used)
        except ImportError as ex:
            expected = str(ex).split()[-1]
            if expected == used:
                raise
            else:
                raise ImportError(f'import error in {used}: {ex}') from ex
        found = annotated_getattr(found, part, used)
    return found

def annotated_getattr(obj: object, name: str, ann: str) -> object:
    if False:
        print('Hello World!')
    try:
        obj = getattr(obj, name)
    except AttributeError as e:
        raise AttributeError('{!r} object at {} has no attribute {!r}'.format(type(obj).__name__, ann, name)) from e
    return obj

def derive_importpath(import_path: str, raising: bool) -> Tuple[str, object]:
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(import_path, str) or '.' not in import_path:
        raise TypeError(f'must be absolute import path string, not {import_path!r}')
    (module, attr) = import_path.rsplit('.', 1)
    target = resolve(module)
    if raising:
        annotated_getattr(target, attr, ann=module)
    return (attr, target)

class Notset:

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return '<notset>'
notset = Notset()

@final
class MonkeyPatch:
    """Helper to conveniently monkeypatch attributes/items/environment
    variables/syspath.

    Returned by the :fixture:`monkeypatch` fixture.

    .. versionchanged:: 6.2
        Can now also be used directly as `pytest.MonkeyPatch()`, for when
        the fixture is not available. In this case, use
        :meth:`with MonkeyPatch.context() as mp: <context>` or remember to call
        :meth:`undo` explicitly.
    """

    def __init__(self) -> None:
        if False:
            return 10
        self._setattr: List[Tuple[object, str, object]] = []
        self._setitem: List[Tuple[Mapping[Any, Any], object, object]] = []
        self._cwd: Optional[str] = None
        self._savesyspath: Optional[List[str]] = None

    @classmethod
    @contextmanager
    def context(cls) -> Generator['MonkeyPatch', None, None]:
        if False:
            for i in range(10):
                print('nop')
        'Context manager that returns a new :class:`MonkeyPatch` object\n        which undoes any patching done inside the ``with`` block upon exit.\n\n        Example:\n\n        .. code-block:: python\n\n            import functools\n\n\n            def test_partial(monkeypatch):\n                with monkeypatch.context() as m:\n                    m.setattr(functools, "partial", 3)\n\n        Useful in situations where it is desired to undo some patches before the test ends,\n        such as mocking ``stdlib`` functions that might break pytest itself if mocked (for examples\n        of this see :issue:`3290`).\n        '
        m = cls()
        try:
            yield m
        finally:
            m.undo()

    @overload
    def setattr(self, target: str, name: object, value: Notset=..., raising: bool=...) -> None:
        if False:
            return 10
        ...

    @overload
    def setattr(self, target: object, name: str, value: object, raising: bool=...) -> None:
        if False:
            print('Hello World!')
        ...

    def setattr(self, target: Union[str, object], name: Union[object, str], value: object=notset, raising: bool=True) -> None:
        if False:
            print('Hello World!')
        '\n        Set attribute value on target, memorizing the old value.\n\n        For example:\n\n        .. code-block:: python\n\n            import os\n\n            monkeypatch.setattr(os, "getcwd", lambda: "/")\n\n        The code above replaces the :func:`os.getcwd` function by a ``lambda`` which\n        always returns ``"/"``.\n\n        For convenience, you can specify a string as ``target`` which\n        will be interpreted as a dotted import path, with the last part\n        being the attribute name:\n\n        .. code-block:: python\n\n            monkeypatch.setattr("os.getcwd", lambda: "/")\n\n        Raises :class:`AttributeError` if the attribute does not exist, unless\n        ``raising`` is set to False.\n\n        **Where to patch**\n\n        ``monkeypatch.setattr`` works by (temporarily) changing the object that a name points to with another one.\n        There can be many names pointing to any individual object, so for patching to work you must ensure\n        that you patch the name used by the system under test.\n\n        See the section :ref:`Where to patch <python:where-to-patch>` in the :mod:`unittest.mock`\n        docs for a complete explanation, which is meant for :func:`unittest.mock.patch` but\n        applies to ``monkeypatch.setattr`` as well.\n        '
        __tracebackhide__ = True
        import inspect
        if isinstance(value, Notset):
            if not isinstance(target, str):
                raise TypeError('use setattr(target, name, value) or setattr(target, value) with target being a dotted import string')
            value = name
            (name, target) = derive_importpath(target, raising)
        elif not isinstance(name, str):
            raise TypeError('use setattr(target, name, value) with name being a string or setattr(target, value) with target being a dotted import string')
        oldval = getattr(target, name, notset)
        if raising and oldval is notset:
            raise AttributeError(f'{target!r} has no attribute {name!r}')
        if inspect.isclass(target):
            oldval = target.__dict__.get(name, notset)
        self._setattr.append((target, name, oldval))
        setattr(target, name, value)

    def delattr(self, target: Union[object, str], name: Union[str, Notset]=notset, raising: bool=True) -> None:
        if False:
            print('Hello World!')
        'Delete attribute ``name`` from ``target``.\n\n        If no ``name`` is specified and ``target`` is a string\n        it will be interpreted as a dotted import path with the\n        last part being the attribute name.\n\n        Raises AttributeError it the attribute does not exist, unless\n        ``raising`` is set to False.\n        '
        __tracebackhide__ = True
        import inspect
        if isinstance(name, Notset):
            if not isinstance(target, str):
                raise TypeError('use delattr(target, name) or delattr(target) with target being a dotted import string')
            (name, target) = derive_importpath(target, raising)
        if not hasattr(target, name):
            if raising:
                raise AttributeError(name)
        else:
            oldval = getattr(target, name, notset)
            if inspect.isclass(target):
                oldval = target.__dict__.get(name, notset)
            self._setattr.append((target, name, oldval))
            delattr(target, name)

    def setitem(self, dic: Mapping[K, V], name: K, value: V) -> None:
        if False:
            print('Hello World!')
        'Set dictionary entry ``name`` to value.'
        self._setitem.append((dic, name, dic.get(name, notset)))
        dic[name] = value

    def delitem(self, dic: Mapping[K, V], name: K, raising: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Delete ``name`` from dict.\n\n        Raises ``KeyError`` if it doesn't exist, unless ``raising`` is set to\n        False.\n        "
        if name not in dic:
            if raising:
                raise KeyError(name)
        else:
            self._setitem.append((dic, name, dic.get(name, notset)))
            del dic[name]

    def setenv(self, name: str, value: str, prepend: Optional[str]=None) -> None:
        if False:
            return 10
        'Set environment variable ``name`` to ``value``.\n\n        If ``prepend`` is a character, read the current environment variable\n        value and prepend the ``value`` adjoined with the ``prepend``\n        character.\n        '
        if not isinstance(value, str):
            warnings.warn(PytestWarning('Value of environment variable {name} type should be str, but got {value!r} (type: {type}); converted to str implicitly'.format(name=name, value=value, type=type(value).__name__)), stacklevel=2)
            value = str(value)
        if prepend and name in os.environ:
            value = value + prepend + os.environ[name]
        self.setitem(os.environ, name, value)

    def delenv(self, name: str, raising: bool=True) -> None:
        if False:
            return 10
        'Delete ``name`` from the environment.\n\n        Raises ``KeyError`` if it does not exist, unless ``raising`` is set to\n        False.\n        '
        environ: MutableMapping[str, str] = os.environ
        self.delitem(environ, name, raising=raising)

    def syspath_prepend(self, path) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Prepend ``path`` to ``sys.path`` list of import locations.'
        if self._savesyspath is None:
            self._savesyspath = sys.path[:]
        sys.path.insert(0, str(path))
        if 'pkg_resources' in sys.modules:
            from pkg_resources import fixup_namespace_packages
            fixup_namespace_packages(str(path))
        from importlib import invalidate_caches
        invalidate_caches()

    def chdir(self, path: Union[str, 'os.PathLike[str]']) -> None:
        if False:
            while True:
                i = 10
        'Change the current working directory to the specified path.\n\n        :param path:\n            The path to change into.\n        '
        if self._cwd is None:
            self._cwd = os.getcwd()
        os.chdir(path)

    def undo(self) -> None:
        if False:
            i = 10
            return i + 15
        'Undo previous changes.\n\n        This call consumes the undo stack. Calling it a second time has no\n        effect unless you do more monkeypatching after the undo call.\n\n        There is generally no need to call `undo()`, since it is\n        called automatically during tear-down.\n\n        .. note::\n            The same `monkeypatch` fixture is used across a\n            single test function invocation. If `monkeypatch` is used both by\n            the test function itself and one of the test fixtures,\n            calling `undo()` will undo all of the changes made in\n            both functions.\n\n            Prefer to use :meth:`context() <pytest.MonkeyPatch.context>` instead.\n        '
        for (obj, name, value) in reversed(self._setattr):
            if value is not notset:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
        self._setattr[:] = []
        for (dictionary, key, value) in reversed(self._setitem):
            if value is notset:
                try:
                    del dictionary[key]
                except KeyError:
                    pass
            else:
                dictionary[key] = value
        self._setitem[:] = []
        if self._savesyspath is not None:
            sys.path[:] = self._savesyspath
            self._savesyspath = None
        if self._cwd is not None:
            os.chdir(self._cwd)
            self._cwd = None