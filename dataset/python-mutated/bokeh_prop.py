""" Thoroughly document Bokeh property attributes.

The ``bokeh-prop`` directive generates documentation for Bokeh model properties,
including cross links to the relevant property types. Additionally, any
per-attribute help strings are also displayed.

This directive takes the name *(class.attr)* of a Bokeh property as its
argument and the module as an option:

.. code-block:: rest

    .. bokeh-prop:: Bar.thing
        :module: bokeh.sphinxext.sample

Examples
--------

For the following definition of ``bokeh.sphinxext.sample.Bar``:

.. code-block:: python

    class Bar(Model):
        ''' This is a Bar model. '''
        thing = List(Int, help="doc for thing")

the above usage yields the output:

    .. bokeh-prop:: Bar.thing
        :module: bokeh.sphinxext.sample


The ``bokeh-prop`` direction may be used explicitly, but it can also be used
in conjunction with the :ref:`bokeh.sphinxext.bokeh_autodoc` extension.

To enable this extension, add `"bokeh.sphinxext.bokeh_prop"` to the
extensions list in your Sphinx configuration module.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import importlib
import textwrap
import warnings
from docutils.parsers.rst.directives import unchanged
from sphinx.errors import SphinxError
from bokeh.core.property._sphinx import type_link
from bokeh.util.warnings import BokehDeprecationWarning
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import PROP_DETAIL
__all__ = ('BokehPropDirective', 'setup')

class BokehPropDirective(BokehDirective):
    has_content = True
    required_arguments = 1
    optional_arguments = 2
    option_spec = {'module': unchanged, 'type': unchanged}

    def run(self):
        if False:
            while True:
                i = 10
        full_name = self.arguments[0]
        (model_name, prop_name) = full_name.rsplit('.')
        module_name = self.options['module']
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            raise SphinxError(f'Could not generate reference docs for {full_name}: could not import module {module_name}')
        model = getattr(module, model_name, None)
        if model is None:
            raise SphinxError(f'Unable to generate reference docs for {full_name}: no model {model_name} in module {module_name}')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=BokehDeprecationWarning)
            model_obj = model()
        try:
            descriptor = model_obj.lookup(prop_name)
        except AttributeError:
            raise SphinxError(f'Unable to generate reference docs for {full_name}: no property {prop_name} on model {model_name}')
        rst_text = PROP_DETAIL.render(name=prop_name, module=self.options['module'], default=repr(descriptor.instance_default(model_obj)), type_info=type_link(descriptor.property), doc='' if descriptor.__doc__ is None else textwrap.dedent(descriptor.__doc__))
        return self.parse(rst_text, '<bokeh-prop>')

def setup(app):
    if False:
        i = 10
        return i + 15
    ' Required Sphinx extension setup function. '
    app.add_directive_to_domain('py', 'bokeh-prop', BokehPropDirective)
    return PARALLEL_SAFE