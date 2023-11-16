""" Thoroughly document Bokeh model classes.

The ``bokeh-model`` directive will automatically document all the attributes
(including Bokeh properties) of a Bokeh Model subclass. A JSON prototype showing
all the possible JSON fields will also be generated.

This directive takes the name of a Bokeh model class as an argument and its
module as an option:

.. code-block:: rest

    .. bokeh-model:: Foo
        :module: bokeh.sphinxext.sample

Examples
--------

For the following definition of ``bokeh.sphinxext.sample.Foo``:

.. code-block:: python

    class Foo(Model):
        ''' This is a Foo model. '''
        index = Either(Auto, Enum('abc', 'def', 'xzy'), help="doc for index")
        value = Tuple(Float, Float, help="doc for value")

usage yields the output:

    .. bokeh-model:: Foo
        :module: bokeh.sphinxext.sample

The ``bokeh-model`` direction may be used explicitly, but it can also be used
in conjunction with the :ref:`bokeh.sphinxext.bokeh_autodoc` extension.

To enable this extension, add `"bokeh.sphinxext.bokeh_model"` to the
extensions list in your Sphinx configuration module.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import importlib
import json
import warnings
from os import getenv
from typing import Any
from docutils.parsers.rst.directives import unchanged
from sphinx.errors import SphinxError
from bokeh.core.property.singletons import Undefined
from bokeh.core.serialization import AnyRep, Serializer, SymbolRep
from bokeh.model import Model
from bokeh.util.warnings import BokehDeprecationWarning
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective, py_sig_re
from .templates import MODEL_DETAIL
__all__ = ('BokehModelDirective', 'setup')

class BokehModelDirective(BokehDirective):
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'module': unchanged, 'canonical': unchanged}

    def run(self):
        if False:
            while True:
                i = 10
        sig = ' '.join(self.arguments)
        m = py_sig_re.match(sig)
        if m is None:
            raise SphinxError(f'Unable to parse signature for bokeh-model: {sig!r}')
        (name_prefix, model_name, arglist, retann) = m.groups()
        if getenv('BOKEH_SPHINX_QUICK') == '1':
            return self.parse(f"{model_name}\n{'-' * len(model_name)}\n", '<bokeh-model>')
        module_name = self.options['module']
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            raise SphinxError(f"Unable to generate model reference docs for {model_name}, couldn't import module {module_name}")
        model = getattr(module, model_name, None)
        if model is None:
            raise SphinxError(f'Unable to generate model reference docs: no model for {model_name} in module {module_name}')
        if not issubclass(model, Model):
            raise SphinxError(f'Unable to generate model reference docs: {model_name}, is not a subclass of Model')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=BokehDeprecationWarning)
            model_obj = model()
        model_json = json.dumps(to_json_rep(model_obj), sort_keys=True, indent=2, separators=(', ', ': '))
        adjusted_module_name = 'bokeh.models' if module_name.startswith('bokeh.models') else module_name
        rst_text = MODEL_DETAIL.render(name=model_name, module_name=adjusted_module_name, model_json=model_json)
        return self.parse(rst_text, '<bokeh-model>')

def setup(app):
    if False:
        i = 10
        return i + 15
    ' Required Sphinx extension setup function. '
    app.add_directive_to_domain('py', 'bokeh-model', BokehModelDirective)
    return PARALLEL_SAFE

class DocsSerializer(Serializer):

    def _encode(self, obj: Any) -> AnyRep:
        if False:
            return 10
        if obj is Undefined:
            return SymbolRep(type='symbol', name='unset')
        else:
            return super()._encode(obj)

def to_json_rep(obj: Model) -> dict[str, AnyRep]:
    if False:
        i = 10
        return i + 15
    serializer = DocsSerializer()
    properties = obj.properties_with_values(include_defaults=True, include_undefined=True)
    attributes = {key: serializer.encode(val) for (key, val) in properties.items()}
    return dict(id=obj.id, **attributes)