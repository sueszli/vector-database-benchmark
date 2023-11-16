from __future__ import annotations
from typing import Any, Sequence, Union
from attrs import define, field
from .._baseclasses import BaseBatch, BaseExtensionType
__all__ = ['PanelView', 'PanelViewArrayLike', 'PanelViewBatch', 'PanelViewLike', 'PanelViewType']

@define(init=False)
class PanelView:
    """
    **Blueprint**: The state of the panels.

    Unstable. Used for the ongoing blueprint experimentations.
    """

    def __init__(self: Any, is_expanded: PanelViewLike):
        if False:
            i = 10
            return i + 15
        'Create a new instance of the PanelView blueprint.'
        self.__attrs_init__(is_expanded=is_expanded)
    is_expanded: bool = field(converter=bool)
PanelViewLike = PanelView
PanelViewArrayLike = Union[PanelView, Sequence[PanelViewLike]]

class PanelViewType(BaseExtensionType):
    _TYPE_NAME: str = 'rerun.blueprint.PanelView'

class PanelViewBatch(BaseBatch[PanelViewArrayLike]):
    _ARROW_TYPE = PanelViewType()