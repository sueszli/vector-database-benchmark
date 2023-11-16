from __future__ import annotations
from typing import Generic, TypeVar, cast
import attr
from .._util import NoPublicConstructor, final
from . import _run
T = TypeVar('T')

@final
class _NoValue:
    ...

@final
@attr.s(eq=False, hash=False, slots=True)
class RunVarToken(Generic[T], metaclass=NoPublicConstructor):
    _var: RunVar[T] = attr.ib()
    previous_value: T | type[_NoValue] = attr.ib(default=_NoValue)
    redeemed: bool = attr.ib(default=False, init=False)

    @classmethod
    def _empty(cls, var: RunVar[T]) -> RunVarToken[T]:
        if False:
            print('Hello World!')
        return cls._create(var)

@final
@attr.s(eq=False, hash=False, slots=True, repr=False)
class RunVar(Generic[T]):
    """The run-local variant of a context variable.

    :class:`RunVar` objects are similar to context variable objects,
    except that they are shared across a single call to :func:`trio.run`
    rather than a single task.

    """
    _name: str = attr.ib()
    _default: T | type[_NoValue] = attr.ib(default=_NoValue)

    def get(self, default: T | type[_NoValue]=_NoValue) -> T:
        if False:
            i = 10
            return i + 15
        'Gets the value of this :class:`RunVar` for the current run call.'
        try:
            return cast(T, _run.GLOBAL_RUN_CONTEXT.runner._locals[self])
        except AttributeError:
            raise RuntimeError('Cannot be used outside of a run context') from None
        except KeyError:
            if default is not _NoValue:
                return default
            if self._default is not _NoValue:
                return self._default
            raise LookupError(self) from None

    def set(self, value: T) -> RunVarToken[T]:
        if False:
            return 10
        'Sets the value of this :class:`RunVar` for this current run\n        call.\n\n        '
        try:
            old_value = self.get()
        except LookupError:
            token = RunVarToken._empty(self)
        else:
            token = RunVarToken[T]._create(self, old_value)
        _run.GLOBAL_RUN_CONTEXT.runner._locals[self] = value
        return token

    def reset(self, token: RunVarToken[T]) -> None:
        if False:
            i = 10
            return i + 15
        'Resets the value of this :class:`RunVar` to what it was\n        previously specified by the token.\n\n        '
        if token is None:
            raise TypeError('token must not be none')
        if token.redeemed:
            raise ValueError('token has already been used')
        if token._var is not self:
            raise ValueError('token is not for us')
        previous = token.previous_value
        try:
            if previous is _NoValue:
                _run.GLOBAL_RUN_CONTEXT.runner._locals.pop(self)
            else:
                _run.GLOBAL_RUN_CONTEXT.runner._locals[self] = previous
        except AttributeError:
            raise RuntimeError('Cannot be used outside of a run context') from None
        token.redeemed = True

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'<RunVar name={self._name!r}>'