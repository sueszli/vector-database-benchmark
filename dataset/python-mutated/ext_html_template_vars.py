from typing import Any, Dict
from sphinx.addnodes import document
from sphinx.application import Sphinx
_CONFIG_VAR = 'html_template_vars'

def update_context(app: Sphinx, pagename: str, templatename: str, context: Dict[str, Any], doctree: document) -> None:
    if False:
        return 10
    for (k, v) in getattr(app.config, _CONFIG_VAR).items():
        context[k] = v

def setup(app: Sphinx) -> None:
    if False:
        i = 10
        return i + 15
    app.add_config_value(_CONFIG_VAR, {}, '')
    app.connect('html-page-context', update_context)