""" Document Bokeh named colors.

The ``bokeh-color`` directive accepts a named color as its argument:

.. code-block:: rest

    .. bokeh-color:: aliceblue

and generates a labeled color swatch as output.

    .. bokeh-color:: aliceblue

The ``bokeh-color`` direction may be used explicitly, but it can also be used
in conjunction with the :ref:`bokeh.sphinxext.bokeh_autodoc` extension.

To enable this extension, add `"bokeh.sphinxext.bokeh_color"` to the extensions
list in your Sphinx configuration module.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from docutils import nodes
from docutils.parsers.rst.directives import unchanged
from bokeh.colors import named
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import COLOR_DETAIL
__all__ = ('BokehColorDirective', 'setup')

class BokehColorDirective(BokehDirective):
    has_content = False
    required_arguments = 1
    option_spec = {'module': unchanged}

    def run(self):
        if False:
            return 10
        color = self.arguments[0]
        html = COLOR_DETAIL.render(color=getattr(named, color).to_css(), text=color)
        node = nodes.raw('', html, format='html')
        return [node]

def setup(app):
    if False:
        while True:
            i = 10
    ' Required Sphinx extension setup function. '
    app.add_directive_to_domain('py', 'bokeh-color', BokehColorDirective)
    return PARALLEL_SAFE