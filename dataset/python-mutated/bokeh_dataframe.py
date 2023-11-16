""" Generate an inline visual representations of a pandas Dataframe.

This directive will embed the output of ``df.head().to_html()`` into the HTML
output.

For example:

.. code-block:: rest

    :bokeh-dataframe:`bokeh.sampledata.sprint.sprint`

Will generate the output:

    :bokeh-dataframe:`bokeh.sampledata.sprint.sprint`

To enable this extension, add `"bokeh.sphinxext.bokeh_dataframe"` to the
extensions list in your Sphinx configuration module.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import importlib
import pandas as pd
from docutils import nodes
from sphinx.errors import SphinxError
from . import PARALLEL_SAFE
__all__ = ('bokeh_dataframe', 'setup')

def bokeh_dataframe(name, rawtext, text, lineno, inliner, options=None, content=None):
    if False:
        print('Hello World!')
    'Generate an inline visual representation of a single color palette.\n\n    If the HTML representation of the dataframe can not be created, a\n    SphinxError is raised to terminate the build.\n\n    For details on the arguments to this function, consult the Docutils docs:\n\n    http://docutils.sourceforge.net/docs/howto/rst-roles.html#define-the-role-function\n\n    '
    (module_name, df_name) = text.rsplit('.', 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise SphinxError(f"Unable to generate HTML table for {df_name}: couldn't import module {module_name}")
    df = getattr(module, df_name, None)
    if df is None:
        raise SphinxError(f'Unable to generate HTML table for {df_name}: no Dataframe {df_name} in module {module_name}')
    if not isinstance(df, pd.DataFrame):
        raise SphinxError(f'{text!r} is not a pandas Dataframe')
    node = nodes.raw('', df.head().to_html(), format='html')
    return ([node], [])

def setup(app):
    if False:
        i = 10
        return i + 15
    ' Required Sphinx extension setup function. '
    app.add_role('bokeh-dataframe', bokeh_dataframe)
    return PARALLEL_SAFE