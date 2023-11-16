""" Various kinds of markup (static content) widgets.

.. warning::
    The explicit purpose of these Bokeh Models is to embed *raw HTML text* for
    a browser to execute. If any portion of the text is derived from untrusted
    user inputs, then you must take appropriate care to sanitize the user input
    prior to passing to Bokeh.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ...core.has_props import abstract
from ...core.properties import Bool, String
from .widget import Widget
__all__ = ('Div', 'Markup', 'Paragraph', 'PreText')

@abstract
class Markup(Widget):
    """ Base class for Bokeh models that represent HTML markup elements.

    Markups include e.g., ``<div>``, ``<p>``, and ``<pre>``.

    Content can be interpreted as `TeX and LaTeX input`_ when rendering as HTML.
    TeX/LaTeX processing can be disabled by setting ``disable_math`` to True.

    .. _`TeX and LaTeX input`: https://docs.mathjax.org/en/latest/basic/mathematics.html#tex-and-latex-input
    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    text = String(default='', help="\n    The text or HTML contents of the widget.\n\n    .. note::\n        If the HTML content contains elements which size depends on\n        on external, asynchronously loaded resources, the size of\n        the widget may be computed incorrectly. This is in particular\n        an issue with images (``<img>``). To remedy this problem, one\n        either has to set explicit dimensions using CSS properties,\n        HTML attributes or model's ``width`` and ``height`` properties,\n        or inline images' contents using data URIs.\n    ")
    disable_math = Bool(False, help='\n    Whether the contents should not be processed as TeX/LaTeX input.\n    ')

class Paragraph(Markup):
    """ A block (paragraph) of text.

    This Bokeh model corresponds to an HTML ``<p>`` element.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    __example__ = 'examples/interaction/widgets/paragraph.py'

class Div(Markup):
    """ A block (div) of text.

    This Bokeh model corresponds to an HTML ``<div>`` element.
    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    __example__ = 'examples/interaction/widgets/div.py'
    render_as_text = Bool(False, help='\n    Whether the contents should be rendered as raw text or as interpreted HTML.\n    The default value is False, meaning contents are rendered as HTML.\n    ')

class PreText(Paragraph):
    """ A block (paragraph) of pre-formatted text.

    This Bokeh model corresponds to an HTML ``<pre>`` element.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    __example__ = 'examples/interaction/widgets/pretext.py'