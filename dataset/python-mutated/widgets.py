from __future__ import annotations
from base64 import b64encode
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Protocol, TypeVar
import reactpy
from reactpy import html
from reactpy._warnings import warn
from reactpy.core.types import ComponentConstructor, VdomDict

def image(format: str, value: str | bytes='', attributes: dict[str, Any] | None=None) -> VdomDict:
    if False:
        return 10
    'Utility for constructing an image from a string or bytes\n\n    The source value will automatically be encoded to base64\n    '
    if format == 'svg':
        format = 'svg+xml'
    if isinstance(value, str):
        bytes_value = value.encode()
    else:
        bytes_value = value
    base64_value = b64encode(bytes_value).decode()
    src = f'data:image/{format};base64,{base64_value}'
    return {'tagName': 'img', 'attributes': {'src': src, **(attributes or {})}}
_Value = TypeVar('_Value')

def use_linked_inputs(attributes: Sequence[dict[str, Any]], on_change: Callable[[_Value], None]=lambda value: None, cast: _CastFunc[_Value]=lambda value: value, initial_value: str='', ignore_empty: bool=True) -> list[VdomDict]:
    if False:
        print('Hello World!')
    'Return a list of linked inputs equal to the number of given attributes.\n\n    Parameters:\n        attributes:\n            That attributes of each returned input element. If the number of generated\n            inputs is variable, you may need to assign each one a\n            :ref:`key <Organizing Items With Keys>` by including a ``"key"`` in each\n            attribute dictionary.\n        on_change:\n            A callback which is triggered when any input is changed. This callback need\n            not update the \'value\' field in the attributes of the inputs since that is\n            handled automatically.\n        cast:\n            Cast the \'value\' of changed inputs that is passed to ``on_change``.\n        initial_value:\n            Initialize the \'value\' field of the inputs.\n        ignore_empty:\n            Do not trigger ``on_change`` if the \'value\' is an empty string.\n    '
    (value, set_value) = reactpy.hooks.use_state(initial_value)

    def sync_inputs(event: dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        new_value = event['target']['value']
        set_value(new_value)
        if not new_value and ignore_empty:
            return None
        on_change(cast(new_value))
    inputs: list[VdomDict] = []
    for attrs in attributes:
        inputs.append(html.input({**attrs, 'on_change': sync_inputs, 'value': value}))
    return inputs
_CastTo_co = TypeVar('_CastTo_co', covariant=True)

class _CastFunc(Protocol[_CastTo_co]):

    def __call__(self, value: str) -> _CastTo_co:
        if False:
            return 10
        ...
if TYPE_CHECKING:
    from reactpy.testing.backend import _MountFunc

def hotswap(update_on_change: bool=False) -> tuple[_MountFunc, ComponentConstructor]:
    if False:
        return 10
    warn("The 'hotswap' function is deprecated and will be removed in a future release", DeprecationWarning, stacklevel=2)
    from reactpy.testing.backend import _hotswap
    return _hotswap(update_on_change)