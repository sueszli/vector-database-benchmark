""" Include plot metadata for plots shown in Bokeh gallery examples.

The ``bokeh-example-metadata`` directive can be used by supplying:

    .. bokeh-example-metadata::
        :sampledata: `sampledata_iris`
        :apis: `~bokeh.plotting.figure.vbar`, :func:`~bokeh.transform.factor_cmap`
        :refs: `ug_basic_bars`
        :words: bar, vbar, legend, factor_cmap, palette

To enable this extension, add `"bokeh.sphinxext.bokeh_example_metadata"` to the
extensions list in your Sphinx configuration module.

"""
from __future__ import annotations
from sphinx.util import logging
log = logging.getLogger(__name__)
from docutils.parsers.rst.directives import unchanged
from sphinx.errors import SphinxError
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import EXAMPLE_METADATA
from .util import get_sphinx_resources
__all__ = ('BokehExampleMetadataDirective', 'setup')
RESOURCES = get_sphinx_resources()

class BokehExampleMetadataDirective(BokehDirective):
    has_content = True
    required_arguments = 0
    option_spec = {'sampledata': unchanged, 'apis': unchanged, 'refs': unchanged, 'keywords': unchanged}

    def run(self):
        if False:
            while True:
                i = 10
        present = self.option_spec.keys() & self.options.keys()
        if not present:
            raise SphinxError('bokeh-example-metadata requires at least one option to be present.')
        extra = self.options.keys() - self.option_spec.keys()
        if extra:
            raise SphinxError(f'bokeh-example-metadata unknown options given: {extra}.')
        rst_text = EXAMPLE_METADATA.render(sampledata=_sampledata(self.options.get('sampledata', None)), apis=_apis(self.options.get('apis', None)), refs=self.options.get('refs', '').split('#')[0], keywords=self.options.get('keywords', '').split('#')[0])
        return self.parse(rst_text, '<bokeh-example-metadata>')

def setup(app):
    if False:
        print('Hello World!')
    ' Required Sphinx extension setup function. '
    app.add_directive('bokeh-example-metadata', BokehExampleMetadataDirective)
    return PARALLEL_SAFE

def _sampledata(mods: str | None) -> str | None:
    if False:
        print('Hello World!')
    if mods is None:
        return
    mods = mods.split('#')[0].strip()
    mods = (mod.strip() for mod in mods.split(','))
    return ', '.join((f':ref:`bokeh.sampledata.{mod} <sampledata_{mod}>`' for mod in mods))

def _apis(apis: str | None) -> str | None:
    if False:
        print('Hello World!')
    if apis is None:
        return
    apis = apis.split('#')[0].strip()
    results = []
    for api in (api.strip() for api in apis.split(',')):
        last = api.split('.')[-1]
        if api.startswith('bokeh.models'):
            results.append(f':class:`bokeh.models.{last} <{api}>`')
        elif 'figure.' in api:
            results.append(f':meth:`figure.{last} <{api}>`')
        elif 'GMap.' in api:
            results.append(f':meth:`GMap.{last} <{api}>`')
        else:
            results.append(f':class:`{api}`')
    return ', '.join(results)