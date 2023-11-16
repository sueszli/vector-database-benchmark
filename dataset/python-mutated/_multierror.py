from __future__ import annotations
import sys
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, cast, overload
import attr
from trio._deprecate import warn_deprecated
if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup, ExceptionGroup
if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing_extensions import Self

def _filter_impl(handler: Callable[[BaseException], BaseException | None], root_exc: BaseException) -> BaseException | None:
    if False:
        for i in range(10):
            print('nop')

    def filter_tree(exc: MultiError | BaseException, preserved: set[int]) -> MultiError | BaseException | None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(exc, MultiError):
            new_exceptions = []
            changed = False
            for child_exc in exc.exceptions:
                new_child_exc = filter_tree(child_exc, preserved)
                if new_child_exc is not child_exc:
                    changed = True
                if new_child_exc is not None:
                    new_exceptions.append(new_child_exc)
            if not new_exceptions:
                return None
            elif changed:
                return MultiError(new_exceptions)
            else:
                preserved.add(id(exc))
                return exc
        else:
            new_exc = handler(exc)
            if new_exc is not None and new_exc is not exc:
                new_exc.__context__ = exc
            return new_exc

    def push_tb_down(tb: TracebackType | None, exc: BaseException, preserved: set[int]) -> None:
        if False:
            return 10
        if id(exc) in preserved:
            return
        new_tb = concat_tb(tb, exc.__traceback__)
        if isinstance(exc, MultiError):
            for child_exc in exc.exceptions:
                push_tb_down(new_tb, child_exc, preserved)
            exc.__traceback__ = None
        else:
            exc.__traceback__ = new_tb
    preserved: set[int] = set()
    new_root_exc = filter_tree(root_exc, preserved)
    push_tb_down(None, root_exc, preserved)
    del filter_tree, push_tb_down
    return new_root_exc

@attr.s(frozen=True)
class MultiErrorCatcher:
    _handler: Callable[[BaseException], BaseException | None] = attr.ib()

    def __enter__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> bool | None:
        if False:
            return 10
        if exc_value is not None:
            filtered_exc = _filter_impl(self._handler, exc_value)
            if filtered_exc is exc_value:
                return False
            if filtered_exc is None:
                return True
            old_context = filtered_exc.__context__
            try:
                raise filtered_exc
            finally:
                (_, value, _) = sys.exc_info()
                assert value is filtered_exc
                value.__context__ = old_context
                del _, filtered_exc, value
        return False
if TYPE_CHECKING:
    _BaseExceptionGroup = BaseExceptionGroup[BaseException]
else:
    _BaseExceptionGroup = BaseExceptionGroup

class MultiError(_BaseExceptionGroup):
    """An exception that contains other exceptions; also known as an
    "inception".

    It's main use is to represent the situation when multiple child tasks all
    raise errors "in parallel".

    Args:
      exceptions (list): The exceptions

    Returns:
      If ``len(exceptions) == 1``, returns that exception. This means that a
      call to ``MultiError(...)`` is not guaranteed to return a
      :exc:`MultiError` object!

      Otherwise, returns a new :exc:`MultiError` object.

    Raises:
      TypeError: if any of the passed in objects are not instances of
          :exc:`BaseException`.

    """

    def __init__(self, exceptions: Sequence[BaseException], *, _collapse: bool=True) -> None:
        if False:
            return 10
        self.collapse = _collapse
        if _collapse and getattr(self, 'exceptions', None) is not None:
            return
        super().__init__('multiple tasks failed', exceptions)

    def __new__(cls, exceptions: Sequence[BaseException], *, _collapse: bool=True) -> NonBaseMultiError | Self | BaseException:
        if False:
            print('Hello World!')
        exceptions = list(exceptions)
        for exc in exceptions:
            if not isinstance(exc, BaseException):
                raise TypeError(f'Expected an exception object, not {exc!r}')
        if _collapse and len(exceptions) == 1:
            return exceptions[0]
        else:
            from_class: type[Self | NonBaseMultiError] = cls
            if all((isinstance(exc, Exception) for exc in exceptions)):
                from_class = NonBaseMultiError
            new_obj = super().__new__(from_class, 'multiple tasks failed', exceptions)
            assert isinstance(new_obj, (cls, NonBaseMultiError))
            return new_obj

    def __reduce__(self) -> tuple[object, tuple[type[Self], list[BaseException]], dict[str, bool]]:
        if False:
            i = 10
            return i + 15
        return (self.__new__, (self.__class__, list(self.exceptions)), {'collapse': self.collapse})

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ', '.join((repr(exc) for exc in self.exceptions))

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'<MultiError: {self}>'

    @overload
    def derive(self, excs: Sequence[Exception], /) -> NonBaseMultiError:
        if False:
            while True:
                i = 10
        ...

    @overload
    def derive(self, excs: Sequence[BaseException], /) -> MultiError:
        if False:
            while True:
                i = 10
        ...

    def derive(self, excs: Sequence[Exception | BaseException], /) -> NonBaseMultiError | MultiError:
        if False:
            print('Hello World!')
        exc = MultiError(excs, _collapse=False)
        exc.collapse = self.collapse
        return exc

    @classmethod
    def filter(cls, handler: Callable[[BaseException], BaseException | None], root_exc: BaseException) -> BaseException | None:
        if False:
            return 10
        'Apply the given ``handler`` to all the exceptions in ``root_exc``.\n\n        Args:\n          handler: A callable that takes an atomic (non-MultiError) exception\n              as input, and returns either a new exception object or None.\n          root_exc: An exception, often (though not necessarily) a\n              :exc:`MultiError`.\n\n        Returns:\n          A new exception object in which each component exception ``exc`` has\n          been replaced by the result of running ``handler(exc)`` – or, if\n          ``handler`` returned None for all the inputs, returns None.\n\n        '
        warn_deprecated('MultiError.filter()', '0.22.0', instead='BaseExceptionGroup.split()', issue=2211)
        return _filter_impl(handler, root_exc)

    @classmethod
    def catch(cls, handler: Callable[[BaseException], BaseException | None]) -> MultiErrorCatcher:
        if False:
            for i in range(10):
                print('nop')
        'Return a context manager that catches and re-throws exceptions\n        after running :meth:`filter` on them.\n\n        Args:\n          handler: as for :meth:`filter`\n\n        '
        warn_deprecated('MultiError.catch', '0.22.0', instead='except* or exceptiongroup.catch()', issue=2211)
        return MultiErrorCatcher(handler)
if TYPE_CHECKING:
    _ExceptionGroup = ExceptionGroup[Exception]
else:
    _ExceptionGroup = ExceptionGroup

class NonBaseMultiError(MultiError, _ExceptionGroup):
    __slots__ = ()
MultiError.__module__ = 'trio'
NonBaseMultiError.__module__ = 'trio'
try:
    import tputil
except ImportError:
    import ctypes
    import _ctypes

    class CTraceback(ctypes.Structure):
        _fields_: ClassVar = [('PyObject_HEAD', ctypes.c_byte * object().__sizeof__()), ('tb_next', ctypes.c_void_p), ('tb_frame', ctypes.c_void_p), ('tb_lasti', ctypes.c_int), ('tb_lineno', ctypes.c_int)]

    def copy_tb(base_tb: TracebackType, tb_next: TracebackType | None) -> TracebackType:
        if False:
            for i in range(10):
                print('nop')
        try:
            raise ValueError
        except ValueError as exc:
            new_tb = exc.__traceback__
            assert new_tb is not None
        c_new_tb = CTraceback.from_address(id(new_tb))
        assert c_new_tb.tb_next is None
        if tb_next is not None:
            _ctypes.Py_INCREF(tb_next)
            c_new_tb.tb_next = id(tb_next)
        assert c_new_tb.tb_frame is not None
        _ctypes.Py_INCREF(base_tb.tb_frame)
        old_tb_frame = new_tb.tb_frame
        c_new_tb.tb_frame = id(base_tb.tb_frame)
        _ctypes.Py_DECREF(old_tb_frame)
        c_new_tb.tb_lasti = base_tb.tb_lasti
        c_new_tb.tb_lineno = base_tb.tb_lineno
        try:
            return new_tb
        finally:
            del new_tb, old_tb_frame
else:

    def copy_tb(base_tb: TracebackType, tb_next: TracebackType | None) -> TracebackType:
        if False:
            while True:
                i = 10

        def controller(operation: tputil.ProxyOperation) -> Any | None:
            if False:
                print('Hello World!')
            if operation.opname in {'__getattribute__', '__getattr__'} and operation.args[0] == 'tb_next':
                return tb_next
            return operation.delegate()
        return cast(TracebackType, tputil.make_proxy(controller, type(base_tb), base_tb))

def concat_tb(head: TracebackType | None, tail: TracebackType | None) -> TracebackType | None:
    if False:
        while True:
            i = 10
    head_tbs = []
    pointer = head
    while pointer is not None:
        head_tbs.append(pointer)
        pointer = pointer.tb_next
    current_head = tail
    for head_tb in reversed(head_tbs):
        current_head = copy_tb(head_tb, tb_next=current_head)
    return current_head
if sys.version_info < (3, 11) and getattr(sys.excepthook, '__name__', None) in ('apport_excepthook', 'partial_apport_excepthook'):
    from types import ModuleType
    import apport_python_hook
    from exceptiongroup import format_exception
    assert sys.excepthook is apport_python_hook.apport_excepthook

    def replacement_excepthook(etype: type[BaseException], value: BaseException, tb: TracebackType | None) -> None:
        if False:
            print('Hello World!')
        sys.stderr.write(''.join(format_exception(etype, value, tb)))
    fake_sys = ModuleType('trio_fake_sys')
    fake_sys.__dict__.update(sys.__dict__)
    fake_sys.__excepthook__ = replacement_excepthook
    apport_python_hook.sys = fake_sys