from __future__ import annotations
from typing import Any, Sequence, Union
import numpy as np
import numpy.typing as npt
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType
from .._converters import to_np_uint8
__all__ = ['ViewportLayout', 'ViewportLayoutArrayLike', 'ViewportLayoutBatch', 'ViewportLayoutLike', 'ViewportLayoutType']

@define(init=False)
class ViewportLayout:
    """
    **Blueprint**: A view of a space.

    Unstable. Used for the ongoing blueprint experimentations.
    """

    def __init__(self: Any, space_view_keys: npt.ArrayLike, tree: npt.ArrayLike, auto_layout: bool):
        if False:
            while True:
                i = 10
        '\n        Create a new instance of the ViewportLayout blueprint.\n\n        Parameters\n        ----------\n        space_view_keys:\n            space_view_keys\n        tree:\n            tree\n        auto_layout:\n            auto_layout\n        '
        self.__attrs_init__(space_view_keys=space_view_keys, tree=tree, auto_layout=auto_layout)
    space_view_keys: npt.NDArray[np.uint8] = field(converter=to_np_uint8)
    tree: npt.NDArray[np.uint8] = field(converter=to_np_uint8)
    auto_layout: bool = field(converter=bool)
ViewportLayoutLike = ViewportLayout
ViewportLayoutArrayLike = Union[ViewportLayout, Sequence[ViewportLayoutLike]]

class ViewportLayoutType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.blueprint.ViewportLayout'

class ViewportLayoutBatch(BaseBatch[ViewportLayoutArrayLike]):
    _ARROW_TYPE = ViewportLayoutType()