""" Various kinds of icons to be used with Button widgets.
See :ref:`ug_interaction_widgets_examples_button` in the |user guide|
for more information.
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ...core.enums import ToolIcon
from ...core.has_props import abstract
from ...core.properties import Color, Either, Enum, FontSize, Int, Required, String
from ...core.property.bases import Init
from ...core.property.singletons import Intrinsic
from ...model import Model
__all__ = ('Icon', 'BuiltinIcon', 'SVGIcon', 'TablerIcon')

@abstract
class Icon(Model):
    """ An abstract base class for icon elements.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    size = Either(Int, FontSize, default='1em', help='\n    The size of the icon. This can be either a number of pixels, or a CSS\n    length string (see https://developer.mozilla.org/en-US/docs/Web/CSS/length).\n    ')

class BuiltinIcon(Icon):
    """ Built-in icons included with BokehJS. """

    def __init__(self, icon_name: Init[str]=Intrinsic, **kwargs) -> None:
        if False:
            return 10
        super().__init__(icon_name=icon_name, **kwargs)
    icon_name = Required(Either(Enum(ToolIcon), String), help='\n    The name of a built-in icon to use. Currently, the following icon names are\n    supported: ``"help"``, ``"question-mark"``, ``"settings"``, ``"x"``\n\n    .. bokeh-plot::\n        :source-position: none\n\n        from bokeh.io import show\n        from bokeh.layouts import column\n        from bokeh.models import BuiltinIcon, Button\n\n        builtin_icons = ["help", "question-mark", "settings", "x"]\n\n        icon_demo = []\n        for icon in builtin_icons:\n            icon_demo.append(Button(label=icon, button_type="light", icon=BuiltinIcon(icon, size="1.2em")))\n\n        show(column(icon_demo))\n\n    ')
    color = Color(default='gray', help='\n    Color to use for the icon.\n    ')

class SVGIcon(Icon):
    """ SVG icons with inline definitions. """

    def __init__(self, svg: Init[str]=Intrinsic, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(svg=svg, **kwargs)
    svg = Required(String, help='\n    The SVG definition of an icon.\n    ')

class TablerIcon(Icon):
    """
    Icons from an external icon provider (https://tabler-icons.io/).

    .. note::
        This icon set is MIT licensed (see https://github.com/tabler/tabler-icons/blob/master/LICENSE).

    .. note::
        External icons are loaded from thrid-party servers and may not be avilable
        immediately (e.g. due to slow iternet connection) or not available at all.
        It isn't possible to create a self-contained bundles with the use of
        ``inline`` resources. To circumvent this, one use ``SVGIcon``, by copying
        the SVG contents of an icon from Tabler's web site.

    """

    def __init__(self, icon_name: Init[str]=Intrinsic, **kwargs) -> None:
        if False:
            return 10
        super().__init__(icon_name=icon_name, **kwargs)
    icon_name = Required(String, help='\n    The name of the icon. See https://tabler-icons.io/ for the list of names.\n    ')