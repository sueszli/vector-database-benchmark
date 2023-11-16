from __future__ import annotations
from typing import Any, Sequence, Union
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType
__all__ = ['AutoSpaceViews', 'AutoSpaceViewsArrayLike', 'AutoSpaceViewsBatch', 'AutoSpaceViewsLike', 'AutoSpaceViewsType']

@define(init=False)
class AutoSpaceViews:
    """
    **Blueprint**: A flag indicating space views should be automatically populated.

    Unstable. Used for the ongoing blueprint experimentations.
    """

    def __init__(self: Any, enabled: AutoSpaceViewsLike):
        if False:
            return 10
        'Create a new instance of the AutoSpaceViews blueprint.'
        self.__attrs_init__(enabled=enabled)
    enabled: bool = field(converter=bool)
AutoSpaceViewsLike = AutoSpaceViews
AutoSpaceViewsArrayLike = Union[AutoSpaceViews, Sequence[AutoSpaceViewsLike]]

class AutoSpaceViewsType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.blueprint.AutoSpaceViews'

class AutoSpaceViewsBatch(BaseBatch[AutoSpaceViewsArrayLike]):
    _ARROW_TYPE = AutoSpaceViewsType()