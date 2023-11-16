from __future__ import annotations
import contextlib
import enum
import os
from collections.abc import Generator
from collections.abc import MutableMapping
from typing import NamedTuple
from typing import Union
_Unset = enum.Enum('_Unset', 'UNSET')
UNSET = _Unset.UNSET

class Var(NamedTuple):
    name: str
    default: str = ''
SubstitutionT = tuple[Union[str, Var], ...]
ValueT = Union[str, _Unset, SubstitutionT]
PatchesT = tuple[tuple[str, ValueT], ...]

def format_env(parts: SubstitutionT, env: MutableMapping[str, str]) -> str:
    if False:
        for i in range(10):
            print('nop')
    return ''.join((env.get(part.name, part.default) if isinstance(part, Var) else part for part in parts))

@contextlib.contextmanager
def envcontext(patch: PatchesT, _env: MutableMapping[str, str] | None=None) -> Generator[None, None, None]:
    if False:
        for i in range(10):
            print('nop')
    'In this context, `os.environ` is modified according to `patch`.\n\n    `patch` is an iterable of 2-tuples (key, value):\n        `key`: string\n        `value`:\n            - string: `environ[key] == value` inside the context.\n            - UNSET: `key not in environ` inside the context.\n            - template: A template is a tuple of strings and Var which will be\n              replaced with the previous environment\n    '
    env = os.environ if _env is None else _env
    before = dict(env)
    for (k, v) in patch:
        if v is UNSET:
            env.pop(k, None)
        elif isinstance(v, tuple):
            env[k] = format_env(v, before)
        else:
            env[k] = v
    try:
        yield
    finally:
        env.clear()
        env.update(before)