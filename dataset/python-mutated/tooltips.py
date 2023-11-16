"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ...core.enums import Anchor, TooltipAttachment
from ...core.properties import Auto, Bool, Either, Enum, Float, Instance, Nullable, Override, Required, String, Tuple
from ..dom import HTML
from ..selectors import Selector
from .ui_element import UIElement
__all__ = ('Tooltip',)

class Tooltip(UIElement):
    """ Render a tooltip.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    visible = Override(default=False)
    position = Nullable(Either(Enum(Anchor), Tuple(Float, Float)), default=None, help='\n    The position of the tooltip with respect to its parent. It can be either\n    an absolute position within the parent or an anchor point for symbolic\n    positioning.\n    ')
    target = Either(Instance(UIElement), Instance(Selector), Auto, default='auto', help='\n    Tooltip can be manually attached to a target UI element or a DOM node\n    (referred to by a selector, e.g. CSS selector or XPath), or its\n    attachment can be inferred from its parent in ``"auto"`` mode.\n    ')
    content = Required(Either(String, Instance(HTML)), help="\n    The tooltip's content. Can be a plaintext string or a :class:`~bokeh.models.HTML`\n    object.\n    ")
    attachment = Either(Enum(TooltipAttachment), Auto, default='auto', help='\n    Whether the tooltip should be displayed to the left or right of the cursor\n    position or above or below it, or if it should be automatically placed\n    in the horizontal or vertical dimension.\n    ')
    show_arrow = Bool(default=True, help="\n    Whether tooltip's arrow should be shown.\n    ")
    closable = Bool(default=False, help='\n    Whether to allow dismissing a tooltip by clicking close (x) button. Useful when\n    using this model for persistent tooltips.\n    ')
    interactive = Bool(default=True, help='\n    Whether to allow pointer events on the contents of this tooltip. Depending\n    on the use case, it may be necessary to disable interactions for better\n    user experience. This however will prevent the user from interacting with\n    the contents of this tooltip, e.g. clicking links.\n    ')