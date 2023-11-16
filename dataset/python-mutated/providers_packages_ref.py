from __future__ import annotations
from typing import TYPE_CHECKING
from provider_yaml_utils import load_package_data
if TYPE_CHECKING:
    from sphinx.application import Sphinx

def _on_config_inited(app, config):
    if False:
        while True:
            i = 10
    del app
    jinja_context = getattr(config, 'jinja_contexts', None) or {}
    jinja_context['providers_ctx'] = {'providers': load_package_data()}
    config.jinja_contexts = jinja_context

def setup(app: Sphinx):
    if False:
        print('Hello World!')
    'Setup plugin'
    app.setup_extension('sphinx_jinja')
    app.connect('config-inited', _on_config_inited)
    app.add_crossref_type(directivename='provider', rolename='provider')
    return {'parallel_read_safe': True, 'parallel_write_safe': True}