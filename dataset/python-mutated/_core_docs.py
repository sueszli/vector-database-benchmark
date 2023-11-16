import os
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, ItemsView, Iterable, Iterator, KeysView, Mapping, MutableMapping, Sequence, ValuesView
from functools import reduce
from types import TracebackType
from typing import IO, Any, Generic, TypeVar, overload
from .docs_argspec import docs_argspec
_save_name = __name__
__name__ = ''
T = TypeVar('T')
KT = TypeVar('KT')
VT = TypeVar('VT')
Tco = TypeVar('Tco', covariant=True)
Vco = TypeVar('Vco', covariant=True)
VTco = TypeVar('VTco', covariant=True)
Tcontra = TypeVar('Tcontra', contravariant=True)
if 'IN_PYTEST' not in os.environ:
    __name__ = 'pyodide.ffi'
_js_flags: dict[str, int] = {}

def _binor_reduce(l: Iterable[int]) -> int:
    if False:
        for i in range(10):
            print('nop')
    return reduce(lambda x, y: x | y, l)

def _process_flag_expression(e: str) -> int:
    if False:
        for i in range(10):
            print('nop')
    return _binor_reduce((_js_flags[x.strip()] for x in e.split('|')))

class _JsProxyMetaClass(type):

    def __instancecheck__(cls, instance):
        if False:
            for i in range(10):
                print('nop')
        'Override for isinstance(instance, cls).'
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass):
        if False:
            return 10
        if type.__subclasscheck__(cls, subclass):
            return True
        if not hasattr(subclass, '_js_type_flags'):
            return False
        cls_flags = cls._js_type_flags
        if isinstance(cls_flags, int):
            cls_flags = [cls_flags]
        else:
            cls_flags = [_process_flag_expression(f) for f in cls_flags]
        subclass_flags = subclass._js_type_flags
        if not isinstance(subclass_flags, int):
            subclass_flags = _binor_reduce((_js_flags[f] for f in subclass_flags))
        return any((cls_flag & subclass_flags == cls_flag for cls_flag in cls_flags))
_instantiate_token = object()

class JsProxy(metaclass=_JsProxyMetaClass):
    """A proxy to make a JavaScript object behave like a Python object

    For more information see the :ref:`type-translations` documentation. In
    particular, see
    :ref:`the list of __dunder__ methods <type-translations-jsproxy>`
    that are (conditionally) implemented on :py:class:`JsProxy`.
    """
    _js_type_flags: Any = 0

    def __new__(cls, arg=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if arg is _instantiate_token:
            return super().__new__(cls)
        raise TypeError(f'{cls.__name__} cannot be instantiated.')

    @property
    def js_id(self) -> int:
        if False:
            while True:
                i = 10
        'An id number which can be used as a dictionary/set key if you want to\n        key on JavaScript object identity.\n\n        If two ``JsProxy`` are made with the same backing JavaScript object, they\n        will have the same ``js_id``.\n        '
        return 0

    @property
    def typeof(self) -> str:
        if False:
            print('Hello World!')
        'Returns the JavaScript type of the ``JsProxy``.\n\n        Corresponds to `typeof obj;` in JavaScript. You may also be interested\n        in the `constuctor` attribute which returns the type as an object.\n        '
        return 'object'

    def object_entries(self) -> 'JsProxy':
        if False:
            i = 10
            return i + 15
        '\n        The JavaScript API ``Object.entries(object)``\n\n        Examples\n        --------\n        >>> from pyodide.code import run_js\n        >>> js_obj = run_js("({first: \'aa\', second: 22})")\n        >>> entries = js_obj.object_entries()\n        >>> [(key, val) for key, val in entries]\n        [(\'first\', \'aa\'), (\'second\', 22)]\n        '
        raise NotImplementedError

    def object_keys(self) -> 'JsProxy':
        if False:
            i = 10
            return i + 15
        '\n        The JavaScript API ``Object.keys(object)``\n\n        Examples\n        --------\n        >>> from pyodide.code import run_js\n        >>> js_obj = run_js("({first: 1, second: 2, third: 3})") # doctest: +SKIP\n        >>> keys = js_obj.object_keys() # doctest: +SKIP\n        >>> list(keys) # doctest: +SKIP\n        [\'first\', \'second\', \'third\']\n        '
        raise NotImplementedError

    def object_values(self) -> 'JsProxy':
        if False:
            i = 10
            return i + 15
        '\n        The JavaScript API ``Object.values(object)``\n\n        Examples\n        --------\n        >>> from pyodide.code import run_js\n        >>> js_obj = run_js("({first: 1, second: 2, third: 3})") # doctest: +SKIP\n        >>> values = js_obj.object_values() # doctest: +SKIP\n        >>> list(values) # doctest: +SKIP\n        [1, 2, 3]\n        '
        raise NotImplementedError

    def as_object_map(self, *, hereditary: bool=False) -> 'JsMutableMap[str, Any]':
        if False:
            while True:
                i = 10
        'Returns a new JsProxy that treats the object as a map.\n\n        The methods :py:func:`~operator.__getitem__`,\n        :py:func:`~operator.__setitem__`, :py:func:`~operator.__contains__`,\n        :py:meth:`~object.__len__`, etc will perform lookups via ``object[key]``\n        or similar.\n\n        Note that ``len(x.as_object_map())`` evaluates in O(n) time (it iterates\n        over the object and counts how many :js:func:`~Reflect.ownKeys` it has).\n        If you need to compute the length in O(1) time, use a real\n        :js:class:`Map` instead.\n\n        Parameters\n        ----------\n        hereditary:\n            If ``True``, any "plain old objects" stored as values in the object\n            will be wrapped in `as_object_map` themselves.\n\n        Examples\n        --------\n\n        .. code-block:: python\n\n            from pyodide.code import run_js\n\n            o = run_js("({x : {y: 2}})")\n            # You have to access the properties of o as attributes\n            assert o.x.y == 2\n            with pytest.raises(TypeError):\n                o["x"] # is not subscriptable\n\n            # as_object_map allows us to access the property with getitem\n            assert o.as_object_map()["x"].y == 2\n\n            with pytest.raises(TypeError):\n                # The inner object is not subscriptable because hereditary is False.\n                o.as_object_map()["x"]["y"]\n\n            # When hereditary is True, the inner object is also subscriptable\n            assert o.as_object_map(hereditary=True)["x"]["y"] == 2\n\n        '
        raise NotImplementedError

    def new(self, *args: Any, **kwargs: Any) -> 'JsProxy':
        if False:
            i = 10
            return i + 15
        'Construct a new instance of the JavaScript object'
        raise NotImplementedError

    def to_py(self, *, depth: int=-1, default_converter: Callable[['JsProxy', Callable[['JsProxy'], Any], Callable[['JsProxy', Any], None]], Any] | None=None) -> Any:
        if False:
            return 10
        'Convert the :class:`JsProxy` to a native Python object as best as\n        possible.\n\n        See :ref:`type-translations-jsproxy-to-py` for more information.\n\n        Parameters\n        ----------\n        depth:\n            Limit the depth of the conversion. If a shallow conversion is\n            desired, set ``depth`` to 1.\n\n        default_converter:\n\n            If present, this will be invoked whenever Pyodide does not have some\n            built in conversion for the object. If ``default_converter`` raises\n            an error, the error will be allowed to propagate. Otherwise, the\n            object returned will be used as the conversion.\n            ``default_converter`` takes three arguments. The first argument is\n            the value to be converted.\n\n        Examples\n        --------\n\n        Here are a couple examples of converter functions. In addition to the\n        normal conversions, convert :js:class:`Date` to :py:class:`~datetime.datetime`:\n\n        .. code-block:: python\n\n            from datetime import datetime\n            def default_converter(value, _ignored1, _ignored2):\n                if value.constructor.name == "Date":\n                    return datetime.fromtimestamp(d.valueOf()/1000)\n                return value\n\n        Don\'t create any JsProxies, require a complete conversion or raise an error:\n\n        .. code-block:: python\n\n            def default_converter(_value, _ignored1, _ignored2):\n                raise Exception("Failed to completely convert object")\n\n        The second and third arguments are only needed for converting\n        containers. The second argument is a conversion function which is used\n        to convert the elements of the container with the same settings. The\n        third argument is a "cache" function which is needed to handle self\n        referential containers. Consider the following example. Suppose we have\n        a Javascript ``Pair`` class:\n\n        .. code-block:: javascript\n\n            class Pair {\n                constructor(first, second){\n                    this.first = first;\n                    this.second = second;\n                }\n            }\n\n        We can use the following ``default_converter`` to convert ``Pair`` to :py:class:`list`:\n\n        .. code-block:: python\n\n            def default_converter(value, convert, cache):\n                if value.constructor.name != "Pair":\n                    return value\n                result = []\n                cache(value, result);\n                result.append(convert(value.first))\n                result.append(convert(value.second))\n                return result\n\n        Note that we have to cache the conversion of ``value`` before converting\n        ``value.first`` and ``value.second``. To see why, consider a self\n        referential pair:\n\n        .. code-block:: javascript\n\n            let p = new Pair(0, 0);\n            p.first = p;\n\n        Without ``cache(value, result);``, converting ``p`` would lead to an\n        infinite recurse. With it, we can successfully convert ``p`` to a list\n        such that ``l[0] is l``.\n        '
        raise NotImplementedError

class JsDoubleProxy(JsProxy):
    """A double proxy created with :py:func:`create_proxy`."""
    _js_type_flags = ['IS_DOUBLE_PROXY']

    def destroy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Destroy the proxy.'
        pass

    def unwrap(self) -> Any:
        if False:
            print('Hello World!')
        'Unwrap a double proxy created with :py:func:`create_proxy` into the\n        wrapped Python object.\n        '
        raise NotImplementedError

class JsPromise(JsProxy):
    """A :py:class:`~pyodide.ffi.JsProxy` of a :js:class:`Promise` or some other `thenable
    <https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise#thenables>`_
    JavaScript object.

    A JavaScript object is considered to be a :js:class:`Promise` if it has a ``then`` method.
    """
    _js_type_flags = ['IS_AWAITABLE']

    def then(self, onfulfilled: Callable[[Any], Any], onrejected: Callable[[Any], Any]) -> 'JsPromise':
        if False:
            print('Hello World!')
        'The :js:meth:`Promise.then` API, wrapped to manage the lifetimes of the\n        handlers.\n\n        Pyodide will automatically release the references to the handlers\n        when the promise resolves.\n        '
        raise NotImplementedError

    def catch(self, onrejected: Callable[[Any], Any], /) -> 'JsPromise':
        if False:
            for i in range(10):
                print('nop')
        'The :js:meth:`Promise.catch` API, wrapped to manage the lifetimes of the\n        handler.\n\n        Pyodide will automatically release the references to the handler\n        when the promise resolves.\n        '
        raise NotImplementedError

    def finally_(self, onfinally: Callable[[], Any], /) -> 'JsPromise':
        if False:
            for i in range(10):
                print('nop')
        'The :js:meth:`Promise.finally` API, wrapped to manage the lifetimes of\n        the handler.\n\n        Pyodide will automatically release the references to the handler\n        when the promise resolves. Note the trailing underscore in the name;\n        this is needed because ``finally`` is a reserved keyword in Python.\n        '
        raise NotImplementedError

class JsBuffer(JsProxy):
    """A JsProxy of an array buffer or array buffer view"""
    _js_type_flags = ['IS_BUFFER']

    def assign(self, rhs: Any, /) -> None:
        if False:
            i = 10
            return i + 15
        'Assign from a Python buffer into the JavaScript buffer.'

    def assign_to(self, to: Any, /) -> None:
        if False:
            print('Hello World!')
        'Assign to a Python buffer from the JavaScript buffer.'

    def to_memoryview(self) -> memoryview:
        if False:
            print('Hello World!')
        'Convert a buffer to a memoryview.\n\n        Copies the data once. This currently has the same effect as\n        :py:meth:`~JsArray.to_py`.\n        '
        raise NotImplementedError

    def to_bytes(self) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        'Convert a buffer to a bytes object.\n\n        Copies the data once.\n        '
        raise NotImplementedError

    def to_file(self, file: IO[bytes] | IO[str], /) -> None:
        if False:
            return 10
        "Writes a buffer to a file.\n\n        Will write the entire contents of the buffer to the current position of\n        the file.\n\n        Example\n        -------\n        >>> import pytest; pytest.skip()\n        >>> from js import Uint8Array\n        >>> x = Uint8Array.new(range(10))\n        >>> with open('file.bin', 'wb') as fh:\n        ...    x.to_file(fh)\n        which is equivalent to,\n        >>> with open('file.bin', 'wb') as fh:\n        ...    data = x.to_bytes()\n        ...    fh.write(data)\n        but the latter copies the data twice whereas the former only copies the\n        data once.\n        "

    def from_file(self, file: IO[bytes] | IO[str], /) -> None:
        if False:
            return 10
        "Reads from a file into a buffer.\n\n        Will try to read a chunk of data the same size as the buffer from\n        the current position of the file.\n\n        Example\n        -------\n        >>> import pytest; pytest.skip()\n        >>> from js import Uint8Array\n        >>> # the JsProxy need to be pre-allocated\n        >>> x = Uint8Array.new(range(10))\n        >>> with open('file.bin', 'rb') as fh:\n        ...    x.read_file(fh)\n        which is equivalent to\n        >>> x = Uint8Array.new(range(10))\n        >>> with open('file.bin', 'rb') as fh:\n        ...    chunk = fh.read(size=x.byteLength)\n        ...    x.assign(chunk)\n        but the latter copies the data twice whereas the former only copies the\n        data once.\n        "

    def _into_file(self, file: IO[bytes] | IO[str], /) -> None:
        if False:
            i = 10
            return i + 15
        "Will write the entire contents of a buffer into a file using\n        ``canOwn : true`` without any copy. After this, the buffer cannot be\n        used again.\n\n        If ``file`` is not empty, its contents will be overwritten!\n\n        Only ``MEMFS`` cares about the ``canOwn`` flag, other file systems will\n        just ignore it.\n\n\n        Example\n        -------\n        >>> import pytest; pytest.skip()\n        >>> from js import Uint8Array\n        >>> x = Uint8Array.new(range(10))\n        >>> with open('file.bin', 'wb') as fh:\n        ...    x._into_file(fh)\n        which is similar to\n        >>> with open('file.bin', 'wb') as fh:\n        ...    data = x.to_bytes()\n        ...    fh.write(data)\n        but the latter copies the data once whereas the former doesn't copy the\n        data.\n        "

    def to_string(self, encoding: str | None=None) -> str:
        if False:
            return 10
        'Convert a buffer to a string object.\n\n        Copies the data twice.\n\n        The encoding argument will be passed to the :js:class:`TextDecoder`\n        constructor. It should be one of the encodings listed in `the table here\n        <https://encoding.spec.whatwg.org/#names-and-labels>`_. The default\n        encoding is utf8.\n        '
        raise NotImplementedError

class JsIterator(JsProxy, Generic[Tco]):
    """A JsProxy of a JavaScript iterator.

    An object is a :py:class:`JsAsyncIterator` if it has a :js:meth:`~Iterator.next` method and either has a
    :js:data:`Symbol.iterator` or has no :js:data:`Symbol.asyncIterator`.
    """
    _js_type_flags = ['IS_ITERATOR']

    def __next__(self) -> Tco:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def __iter__(self) -> Iterator[Tco]:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

class JsAsyncIterator(JsProxy, Generic[Tco]):
    """A JsProxy of a JavaScript async iterator.

    An object is a :py:class:`JsAsyncIterator` if it has a
    :js:meth:`~AsyncIterator.next` method and either has a
    :js:data:`Symbol.asyncIterator` or has no :js:data:`Symbol.iterator`
    """
    _js_type_flags = ['IS_ASYNC_ITERATOR']

    def __anext__(self) -> Awaitable[Tco]:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def __aiter__(self) -> AsyncIterator[Tco]:
        if False:
            while True:
                i = 10
        raise NotImplementedError

class JsIterable(JsProxy, Generic[Tco]):
    """A JavaScript iterable object

    A JavaScript object is iterable if it has a :js:data:`Symbol.iterator` method.
    """
    _js_type_flags = ['IS_ITERABLE']

    def __iter__(self) -> Iterator[Tco]:
        if False:
            print('Hello World!')
        raise NotImplementedError

class JsAsyncIterable(JsProxy, Generic[Tco]):
    """A JavaScript async iterable object

    A JavaScript object is async iterable if it has a :js:data:`Symbol.asyncIterator`
    method.
    """
    _js_type_flags = ['IS_ASYNC_ITERABLE']

    def __aiter__(self) -> AsyncIterator[Tco]:
        if False:
            print('Hello World!')
        raise NotImplementedError

class JsGenerator(JsIterable[Tco], Generic[Tco, Tcontra, Vco]):
    """A JavaScript generator

    A JavaScript object is treated as a generator if its
    :js:data:`Symbol.toStringTag` is ``"Generator"``. Most likely this will be
    because it is a true :js:class:`Generator` produced by the JavaScript
    runtime, but it may be a custom object trying hard to pretend to be a
    generator. It should have :js:meth:`~Generator.next`,
    :js:meth:`~Generator.return` and :js:meth:`~Generator.throw` methods.
    """
    _js_type_flags = ['IS_GENERATOR']

    def send(self, value: Tcontra) -> Tco:
        if False:
            return 10
        '\n        Resumes the execution and "sends" a value into the generator function.\n\n        The ``value`` argument becomes the result of the current yield\n        expression. The ``send()`` method returns the next value yielded by the\n        generator, or raises :py:exc:`StopIteration` if the generator exits without\n        yielding another value. When ``send()`` is called to start the\n        generator, the argument will be ignored. Unlike in Python, we cannot\n        detect that the generator hasn\'t started yet, and no error will be\n        thrown if the argument of a not-started generator is not ``None``.\n        '
        raise NotImplementedError

    @overload
    def throw(self, typ: type[BaseException], val: BaseException | object=..., tb: TracebackType | None=..., /) -> Tco:
        if False:
            return 10
        ...

    @overload
    def throw(self, typ: BaseException, val: None=..., tb: TracebackType | None=..., /) -> Tco:
        if False:
            for i in range(10):
                print('nop')
        ...

    @docs_argspec('(self, error: BaseException, /) -> Tco')
    def throw(self, *args: Any) -> Tco:
        if False:
            return 10
        '\n        Raises an exception at the point where the generator was paused, and\n        returns the next value yielded by the generator function.\n\n        If the generator exits without yielding another value, a\n        :py:exc:`StopIteration` exception is raised. If the generator function does\n        not catch the passed-in exception, or raises a different exception, then\n        that exception propagates to the caller.\n\n        In typical use, this is called with a single exception instance similar\n        to the way the raise keyword is used.\n\n        For backwards compatibility, however, a second signature is supported,\n        following a convention from older versions of Python. The type argument\n        should be an exception class, and value should be an exception instance.\n        If the value is not provided, the type constructor is called to get an\n        instance. If traceback is provided, it is set on the exception,\n        otherwise any existing ``__traceback__`` attribute stored in value may\n        be cleared.\n        '
        raise NotImplementedError

    def close(self) -> None:
        if False:
            while True:
                i = 10
        'Raises a :py:exc:`GeneratorExit` at the point where the generator\n        function was paused.\n\n        If the generator function then exits gracefully, is already closed, or\n        raises :py:exc:`GeneratorExit` (by not catching the exception), ``close()``\n        returns to its caller. If the generator yields a value, a\n        :py:exc:`RuntimeError` is raised. If the generator raises any other\n        exception, it is propagated to the caller. ``close()`` does nothing if\n        the generator has already exited due to an exception or normal exit.\n        '

    def __next__(self) -> Tco:
        if False:
            return 10
        raise NotImplementedError

    def __iter__(self) -> 'JsGenerator[Tco, Tcontra, Vco]':
        if False:
            print('Hello World!')
        raise NotImplementedError

class JsFetchResponse(JsProxy):
    """A :py:class:`JsFetchResponse` object represents a :js:data:`Response` to a
    :js:func:`fetch` request.
    """
    bodyUsed: bool
    ok: bool
    redirected: bool
    status: int
    statusText: str
    type: str
    url: str
    headers: Any

    def clone(self) -> 'JsFetchResponse':
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    async def arrayBuffer(self) -> JsBuffer:
        raise NotImplementedError

    async def text(self) -> str:
        raise NotImplementedError

    async def json(self) -> JsProxy:
        raise NotImplementedError

class JsAsyncGenerator(JsAsyncIterable[Tco], Generic[Tco, Tcontra, Vco]):
    """A JavaScript :js:class:`AsyncGenerator`

    A JavaScript object is treated as an async generator if it's
    :js:data:`Symbol.toStringTag` is ``"AsyncGenerator"``. Most likely this will
    be because it is a true async generator produced by the JavaScript runtime,
    but it may be a custom object trying hard to pretend to be an async
    generator. It should have :js:meth:`~AsyncGenerator.next`,
    :js:meth:`~AsyncGenerator.return`, and :js:meth:`~AsyncGenerator.throw`
    methods.
    """
    _js_type_flags = ['IS_ASYNC_GENERATOR']

    def __anext__(self) -> Awaitable[Tco]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def __aiter__(self) -> 'JsAsyncGenerator[Tco, Tcontra, Vco]':
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def asend(self, value: Tcontra, /) -> Awaitable[Tco]:
        if False:
            for i in range(10):
                print('nop')
        'Resumes the execution and "sends" a value into the async generator\n        function.\n\n        The ``value`` argument becomes the result of the current yield\n        expression. The awaitable returned by the ``asend()`` method will return\n        the next value yielded by the generator or raises\n        :py:exc:`StopAsyncIteration` if the asynchronous generator returns. If the\n        generator returned a value, this value is discarded (because in Python\n        async generators cannot return a value).\n\n        When ``asend()`` is called to start the generator, the argument will be\n        ignored. Unlike in Python, we cannot detect that the generator hasn\'t\n        started yet, and no error will be thrown if the argument of a\n        not-started generator is not ``None``.\n        '
        raise NotImplementedError

    @overload
    def athrow(self, typ: type[BaseException], val: BaseException | object=..., tb: TracebackType | None=..., /) -> Awaitable[Tco]:
        if False:
            return 10
        ...

    @overload
    def athrow(self, typ: BaseException, val: None=..., tb: TracebackType | None=..., /) -> Awaitable[Tco]:
        if False:
            while True:
                i = 10
        ...

    @docs_argspec('(self, error: BaseException, /) -> Tco')
    def athrow(self, value: Any, *args: Any) -> Awaitable[Tco]:
        if False:
            for i in range(10):
                print('nop')
        'Resumes the execution and raises an exception at the point where the\n        generator was paused.\n\n        The awaitable returned by ``athrow()`` method will return the next value\n        yielded by the generator or raises :py:exc:`StopAsyncIteration` if the\n        asynchronous generator returns. If the generator returned a value, this\n        value is discarded (because in Python async generators cannot return a\n        value). If the generator function does not catch the passed-in\n        exception, or raises a different exception, then that exception\n        propagates to the caller.\n        '
        raise NotImplementedError

    def aclose(self) -> Awaitable[None]:
        if False:
            i = 10
            return i + 15
        'Raises a :py:exc:`GeneratorExit` at the point where the generator\n        function was paused.\n\n        If the generator function then exits gracefully, is already closed, or\n        raises :py:exc:`GeneratorExit` (by not catching the exception),\n        ``aclose()`` returns to its caller. If the generator yields a value, a\n        :py:exc:`RuntimeError` is raised. If the generator raises any other\n        exception, it is propagated to the caller. ``aclose()`` does nothing if\n        the generator has already exited due to an exception or normal exit.\n        '
        raise NotImplementedError

class JsCallable(JsProxy):
    _js_type_flags = ['IS_CALLABLE']

    def __call__(self):
        if False:
            return 10
        pass

class JsArray(JsIterable[T], Generic[T]):
    """A JsProxy of an :js:class:`Array`, :js:class:`NodeList`, or :js:class:`TypedArray`"""
    _js_type_flags = ['IS_ARRAY', 'IS_NODE_LIST', 'IS_TYPEDARRAY']

    def __getitem__(self, idx: int | slice) -> T:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def __setitem__(self, idx: int | slice, value: T) -> None:
        if False:
            print('Hello World!')
        pass

    def __delitem__(self, idx: int | slice) -> None:
        if False:
            return 10
        pass

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 0

    def extend(self, other: Iterable[T], /) -> None:
        if False:
            return 10
        'Extend array by appending elements from the iterable.'

    def __reversed__(self) -> Iterator[T]:
        if False:
            for i in range(10):
                print('nop')
        'Return a reverse iterator over the :js:class:`Array`.'
        raise NotImplementedError

    def pop(self, /, index: int=-1) -> T:
        if False:
            while True:
                i = 10
        'Remove and return the ``item`` at ``index`` (default last).\n\n        Raises :py:exc:`IndexError` if list is empty or index is out of range.\n        '
        raise NotImplementedError

    def push(self, /, object: T) -> None:
        if False:
            return 10
        pass

    def append(self, /, object: T) -> None:
        if False:
            while True:
                i = 10
        'Append object to the end of the list.'

    def index(self, /, value: T, start: int=0, stop: int=sys.maxsize) -> int:
        if False:
            while True:
                i = 10
        'Return first ``index`` at which ``value`` appears in the ``Array``.\n\n        Raises :py:exc:`ValueError` if the value is not present.\n        '
        raise NotImplementedError

    def count(self, /, x: T) -> int:
        if False:
            while True:
                i = 10
        'Return the number of times x appears in the list.'
        raise NotImplementedError

    def reverse(self) -> None:
        if False:
            print('Hello World!')
        'Reverse the array in place.\n\n        Present only if the wrapped Javascript object is an array.\n        '

    def to_py(self, *, depth: int=-1, default_converter: Callable[['JsProxy', Callable[['JsProxy'], Any], Callable[['JsProxy', Any], None]], Any] | None=None) -> list[Any]:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def __mul__(self, other: int) -> 'JsArray[T]':
        if False:
            while True:
                i = 10
        raise NotImplementedError

class JsTypedArray(JsBuffer, JsArray[int]):
    _js_type_flags = ['IS_TYPEDARRAY']
    BYTES_PER_ELEMENT: int

    def subarray(self, start: int | None=None, stop: int | None=None) -> 'JsTypedArray':
        if False:
            print('Hello World!')
        raise NotImplementedError
    buffer: JsBuffer

@Mapping.register
class JsMap(JsIterable[KT], Generic[KT, VTco]):
    """A JavaScript Map

    To be considered a map, a JavaScript object must have a ``get`` method, it
    must have a ``size`` or a ``length`` property which is a number
    (idiomatically it should be called ``size``) and it must be iterable.
    """
    _js_type_flags = ['HAS_GET | HAS_LENGTH | IS_ITERABLE', 'IS_OBJECT_MAP']

    def __getitem__(self, idx: KT) -> VTco:
        if False:
            return 10
        raise NotImplementedError

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return 0

    def __contains__(self, idx: KT) -> bool:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def keys(self) -> KeysView[KT]:
        if False:
            while True:
                i = 10
        'Return a :py:class:`~collections.abc.KeysView` for the map.'
        raise NotImplementedError

    def items(self) -> ItemsView[KT, VTco]:
        if False:
            while True:
                i = 10
        'Return a :py:class:`~collections.abc.ItemsView` for the map.'
        raise NotImplementedError

    def values(self) -> ValuesView[VTco]:
        if False:
            print('Hello World!')
        'Return a :py:class:`~collections.abc.ValuesView` for the map.'
        raise NotImplementedError

    @overload
    def get(self, key: KT, /) -> VTco | None:
        if False:
            while True:
                i = 10
        ...

    @overload
    def get(self, key: KT, default: VTco | T, /) -> VTco | T:
        if False:
            while True:
                i = 10
        ...

    @docs_argspec('(self, key: KT, default: VTco | None, /) -> VTco')
    def get(self, key: KT, default: Any=None, /) -> VTco:
        if False:
            print('Hello World!')
        'If ``key in self``, returns ``self[key]``. Otherwise returns ``default``.'
        raise NotImplementedError

@MutableMapping.register
class JsMutableMap(JsMap[KT, VT], Generic[KT, VT]):
    """A JavaScript mutable map

    To be considered a mutable map, a JavaScript object must have a ``get``
    method, a ``has`` method, a ``size`` or a ``length`` property which is a
    number (idiomatically it should be called ``size``) and it must be iterable.

    Instances of the JavaScript builtin ``Map`` class are ``JsMutableMap`` s.
    Also proxies returned by :py:meth:`JsProxy.as_object_map` are instances of
    ``JsMap`` .
    """
    _js_type_flags = ['HAS_GET | HAS_SET | HAS_LENGTH | IS_ITERABLE', 'IS_OBJECT_MAP']

    @overload
    def pop(self, key: KT, /) -> VT:
        if False:
            print('Hello World!')
        ...

    @overload
    def pop(self, key: KT, default: VT | T=..., /) -> VT | T:
        if False:
            return 10
        ...

    @docs_argspec('(self, key: KT, default: VT | None = None, /) -> VT')
    def pop(self, key: KT, default: Any=None, /) -> Any:
        if False:
            return 10
        'If ``key in self``, return ``self[key]`` and remove key from ``self``. Otherwise\n        returns ``default``.\n        '
        raise NotImplementedError

    def setdefault(self, key: KT, default: VT | None=None) -> VT:
        if False:
            while True:
                i = 10
        'If ``key in self``, return ``self[key]``. Otherwise\n        sets ``self[key] = default`` and returns ``default``.\n        '
        raise NotImplementedError

    def popitem(self) -> tuple[KT, VT]:
        if False:
            i = 10
            return i + 15
        'Remove some arbitrary ``key, value`` pair from the map and returns the\n        ``(key, value)`` tuple.\n        '
        raise NotImplementedError

    def clear(self) -> None:
        if False:
            return 10
        'Empty out the map entirely.'

    @overload
    def update(self, __m: Mapping[KT, VT], **kwargs: VT) -> None:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def update(self, __m: Iterable[tuple[KT, VT]], **kwargs: VT) -> None:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def update(self, **kwargs: VT) -> None:
        if False:
            print('Hello World!')
        ...

    @docs_argspec('(self, other : Mapping[KT, VT] | Iterable[tuple[KT, VT]] = None , /, **kwargs) -> None')
    def update(self, *args: Any, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Updates ``self`` from ``other`` and ``kwargs``.\n\n        Parameters\n        ----------\n            other:\n\n                Either a mapping or an iterable of pairs. This can be left out.\n\n            kwargs:  ``VT``\n\n                Extra key-values pairs to insert into the map. Only usable for\n                inserting extra strings.\n\n        If ``other`` is present and is a :py:class:`~collections.abc.Mapping` or has a ``keys``\n        method, does\n\n        .. code-block:: python\n\n            for k in other:\n                self[k] = other[k]\n\n        If ``other`` is present and lacks a ``keys`` method, does\n\n        .. code-block:: python\n\n            for (k, v) in other:\n                self[k] = v\n\n        In all cases this is followed by:\n\n        .. code-block:: python\n\n            for (k, v) in kwargs.items():\n                self[k] = v\n\n        '

    def __setitem__(self, idx: KT, value: VT) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def __delitem__(self, idx: KT) -> None:
        if False:
            i = 10
            return i + 15
        return None

class JsOnceCallable(JsCallable):

    def destroy(self):
        if False:
            return 10
        pass

class JsException(JsProxy, Exception):
    """A JavaScript Error.

    These are pickleable unlike other JsProxies.
    """

    def __new__(cls, *args, **kwargs):
        if False:
            print('Hello World!')
        if args[0] == _instantiate_token:
            return super().__new__(cls, *args, **kwargs)
        return cls._new_exc(*args, **kwargs)

    @classmethod
    def _new_exc(cls, name: str, message: str='', stack: str='') -> 'JsException':
        if False:
            i = 10
            return i + 15
        result = super().__new__(JsException, _instantiate_token)
        result.name = name
        result.message = message
        result.stack = stack
        return result

    @classmethod
    def new(cls, *args: Any) -> 'JsException':
        if False:
            for i in range(10):
                print('nop')
        return cls()

    def __str__(self):
        if False:
            print('Hello World!')
        return f'{self.name}: {self.message}'
    name: str
    'The name of the error type'
    message: str
    'The error message'
    stack: str
    'The JavaScript stack trace'

class ConversionError(Exception):
    """An error thrown when conversion between JavaScript and Python fails."""

class InternalError(Exception):
    """Thrown when a recoverable assertion error occurs in internal Pyodide code"""
    pass

class JsDomElement(JsProxy):
    id: str

    @property
    def tagName(self) -> str:
        if False:
            return 10
        return ''

    @property
    def children(self) -> Sequence['JsDomElement']:
        if False:
            i = 10
            return i + 15
        return []

    def appendChild(self, child: 'JsDomElement') -> None:
        if False:
            return 10
        pass

    def addEventListener(self, event: str, listener: Callable[[Any], None]) -> None:
        if False:
            return 10
        pass

    def removeEventListener(self, event: str, listener: Callable[[Any], None]) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    @property
    def style(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        pass

def create_once_callable(obj: Callable[..., Any], /) -> JsOnceCallable:
    if False:
        print('Hello World!')
    'Wrap a Python Callable in a JavaScript function that can be called once.\n\n    After being called the proxy will decrement the reference count\n    of the Callable. The JavaScript function also has a ``destroy`` API that\n    can be used to release the proxy without calling it.\n    '
    return obj

def create_proxy(obj: Any, /, *, capture_this: bool=False, roundtrip: bool=True) -> JsDoubleProxy:
    if False:
        while True:
            i = 10
    'Create a :py:class:`JsProxy` of a :js:class:`~pyodide.ffi.PyProxy`.\n\n    This allows explicit control over the lifetime of the\n    :js:class:`~pyodide.ffi.PyProxy` from Python: call the\n    :py:meth:`~JsDoubleProxy.destroy` API when done.\n\n    Parameters\n    ----------\n    obj:\n        The object to wrap.\n\n    capture_this :\n        If the object is callable, should ``this`` be passed as the first\n        argument when calling it from JavaScript.\n\n    roundtrip:\n        When the proxy is converted back from JavaScript to Python, if this is\n        ``True`` it is converted into a double proxy. If ``False``, it is\n        unwrapped into a Python object. In the case that ``roundtrip`` is\n        ``True`` it is possible to unwrap a double proxy with the\n        :py:meth:`JsDoubleProxy.unwrap` method. This is useful to allow easier\n        control of lifetimes from Python:\n\n        .. code-block:: python\n\n            from js import o\n            d = {}\n            o.d = create_proxy(d, roundtrip=True)\n            o.d.destroy() # Destroys the proxy created with create_proxy\n\n        With ``roundtrip=False`` this would be an error.\n    '
    return obj

@overload
def to_js(obj: list[Any] | tuple[Any], /, *, depth: int=-1, pyproxies: JsProxy | None=None, create_pyproxies: bool=True, dict_converter: Callable[[Iterable[JsArray[Any]]], JsProxy] | None=None, default_converter: Callable[[Any, Callable[[Any], JsProxy], Callable[[Any, JsProxy], None]], JsProxy] | None=None) -> JsArray[Any]:
    if False:
        while True:
            i = 10
    ...

@overload
def to_js(obj: dict[Any, Any], /, *, depth: int=-1, pyproxies: JsProxy | None, create_pyproxies: bool, dict_converter: None, default_converter: Callable[[Any, Callable[[Any], JsProxy], Callable[[Any, JsProxy], None]], JsProxy] | None=None) -> JsMap[Any, Any]:
    if False:
        return 10
    ...

@overload
def to_js(obj: Any, /, *, depth: int=-1, pyproxies: JsProxy | None=None, create_pyproxies: bool=True, dict_converter: Callable[[Iterable[JsArray[Any]]], JsProxy] | None=None, default_converter: Callable[[Any, Callable[[Any], JsProxy], Callable[[Any, JsProxy], None]], JsProxy] | None=None) -> Any:
    if False:
        i = 10
        return i + 15
    ...

def to_js(obj: Any, /, *, depth: int=-1, pyproxies: JsProxy | None=None, create_pyproxies: bool=True, dict_converter: Callable[[Iterable[JsArray[Any]]], JsProxy] | None=None, default_converter: Callable[[Any, Callable[[Any], JsProxy], Callable[[Any, JsProxy], None]], JsProxy] | None=None) -> Any:
    if False:
        return 10
    'Convert the object to JavaScript.\n\n    This is similar to :js:meth:`~pyodide.ffi.PyProxy.toJs`, but for use from Python. If the\n    object can be implicitly translated to JavaScript, it will be returned\n    unchanged. If the object cannot be converted into JavaScript, this method\n    will return a :py:class:`JsProxy` of a :js:class:`~pyodide.ffi.PyProxy`, as if you had used\n    :func:`~pyodide.ffi.create_proxy`.\n\n    See :ref:`type-translations-pyproxy-to-js` for more information.\n\n    Parameters\n    ----------\n    obj :\n        The Python object to convert\n\n    depth :\n        The maximum depth to do the conversion. Negative numbers are treated as\n        infinite. Set this to 1 to do a shallow conversion.\n\n    pyproxies:\n        Should be a JavaScript :js:class:`Array`. If provided, any ``PyProxies``\n        generated will be stored here. You can later use :py:meth:`destroy_proxies`\n        if you want to destroy the proxies from Python (or from JavaScript you\n        can just iterate over the :js:class:`Array` and destroy the proxies).\n\n    create_pyproxies:\n        If you set this to :py:data:`False`, :py:func:`to_js` will raise an error rather\n        than creating any pyproxies.\n\n    dict_converter:\n        This converter if provided receives a (JavaScript) iterable of\n        (JavaScript) pairs [key, value]. It is expected to return the desired\n        result of the dict conversion. Some suggested values for this argument:\n\n          * ``js.Map.new`` -- similar to the default behavior\n          * ``js.Array.from`` -- convert to an array of entries\n          * ``js.Object.fromEntries`` -- convert to a JavaScript object\n\n    default_converter:\n        If present will be invoked whenever Pyodide does not have some built in\n        conversion for the object. If ``default_converter`` raises an error, the\n        error will be allowed to propagate. Otherwise, the object returned will\n        be used as the conversion. ``default_converter`` takes three arguments.\n        The first argument is the value to be converted.\n\n    Examples\n    --------\n    >>> from js import Object, Map, Array # doctest: +SKIP\n    >>> from pyodide.ffi import to_js # doctest: +SKIP\n    >>> js_object = to_js({\'age\': 20, \'name\': \'john\'}) # doctest: +SKIP\n    >>> js_object # doctest: +SKIP\n    [object Map]\n    >>> js_object.keys(), js_object.values() # doctest: +SKIP\n    KeysView([object Map]) ValuesView([object Map]) # doctest: +SKIP\n    >>> [(k, v) for k, v in zip(js_object.keys(), js_object.values())] # doctest: +SKIP\n    [(\'age\', 20), (\'name\', \'john\')]\n\n    >>> js_object = to_js({\'age\': 20, \'name\': \'john\'}, dict_converter=Object.fromEntries) # doctest: +SKIP\n    >>> js_object.age == 20 # doctest: +SKIP\n    True\n    >>> js_object.name == \'john\' # doctest: +SKIP\n    True\n    >>> js_object # doctest: +SKIP\n    [object Object]\n    >>> js_object.hasOwnProperty("age") # doctest: +SKIP\n    True\n    >>> js_object.hasOwnProperty("height") # doctest: +SKIP\n    False\n\n    >>> js_object = to_js({\'age\': 20, \'name\': \'john\'}, dict_converter=Array.from_) # doctest: +SKIP\n    >>> [item for item in js_object] # doctest: +SKIP\n    [age,20, name,john]\n    >>> js_object.toString() # doctest: +SKIP\n    age,20,name,john\n\n    >>> class Bird: pass # doctest: +SKIP\n    >>> converter = lambda value, convert, cache: Object.new(size=1, color=\'red\') if isinstance(value, Bird) else None # doctest: +SKIP\n    >>> js_nest = to_js([Bird(), Bird()], default_converter=converter) # doctest: +SKIP\n    >>> [bird for bird in js_nest] # doctest: +SKIP\n    [[object Object], [object Object]]\n    >>> [(bird.size, bird.color) for bird in js_nest] # doctest: +SKIP\n    [(1, \'red\'), (1, \'red\')]\n\n    Here are some examples demonstrating the usage of the ``default_converter``\n    argument.\n\n\n    In addition to the normal conversions, convert JavaScript :js:class:`Date`\n    objects to :py:class:`~datetime.datetime` objects:\n\n    .. code-block:: python\n\n        from datetime import datetime\n        from js import Date\n        def default_converter(value, _ignored1, _ignored2):\n            if isinstance(value, datetime):\n                return Date.new(value.timestamp() * 1000)\n            return value\n\n    Don\'t create any PyProxies, require a complete conversion or raise an error:\n\n    .. code-block:: python\n\n        def default_converter(_value, _ignored1, _ignored2):\n            raise Exception("Failed to completely convert object")\n\n    The second and third arguments are only needed for converting containers.\n    The second argument is a conversion function which is used to convert the\n    elements of the container with the same settings. The third argument is a\n    "cache" function which is needed to handle self referential containers.\n    Consider the following example. Suppose we have a Python ``Pair`` class:\n\n    .. code-block:: python\n\n        class Pair:\n            def __init__(self, first, second):\n                self.first = first\n                self.second = second\n\n    We can use the following ``default_converter`` to convert ``Pair`` to\n    :js:class:`Array`:\n\n    .. code-block:: python\n\n        from js import Array\n\n        def default_converter(value, convert, cache):\n            if not isinstance(value, Pair):\n                return value\n            result = Array.new()\n            cache(value, result)\n            result.push(convert(value.first))\n            result.push(convert(value.second))\n            return result\n\n    Note that we have to cache the conversion of ``value`` before converting\n    ``value.first`` and ``value.second``. To see why, consider a self\n    referential pair:\n\n    .. code-block:: javascript\n\n        p = Pair(0, 0); p.first = p;\n\n    Without ``cache(value, result);``, converting ``p`` would lead to an\n    infinite recurse. With it, we can successfully convert ``p`` to an Array\n    such that ``l[0] === l``.\n    '
    return obj

def destroy_proxies(pyproxies: JsArray[Any], /) -> None:
    if False:
        i = 10
        return i + 15
    'Destroy all PyProxies in a JavaScript array.\n\n    pyproxies must be a JavaScript Array of PyProxies. Intended for use\n    with the arrays created from the "pyproxies" argument of :js:meth:`~pyodide.ffi.PyProxy.toJs`\n    and :py:func:`to_js`. This method is necessary because indexing the Array from\n    Python automatically unwraps the PyProxy into the wrapped Python object.\n    '
    pass
__name__ = _save_name
del _save_name
__all__ = ['ConversionError', 'InternalError', 'JsArray', 'JsAsyncGenerator', 'JsAsyncIterable', 'JsAsyncIterator', 'JsBuffer', 'JsDoubleProxy', 'JsException', 'JsFetchResponse', 'JsGenerator', 'JsIterable', 'JsIterator', 'JsMap', 'JsMutableMap', 'JsPromise', 'JsProxy', 'JsDomElement', 'JsCallable', 'JsTypedArray', 'create_once_callable', 'create_proxy', 'destroy_proxies', 'to_js']