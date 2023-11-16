"""A sphinx extension to process jinja/rst template

Usage:
    define the context variable needed by the document inside
    ``jinja_contexts`` variable in ``conf.py``
"""
from pathlib import Path
import jinja2
from . import rst_helpers, utils

def rstjinja(app, docname, source):
    if False:
        for i in range(10):
            print('nop')
    '\n    Render our pages as a jinja template for fancy templating goodness.\n    '
    if app.builder.format != 'html':
        return
    print(docname)
    page_ctx = app.config.jinja_contexts.get(docname)
    if page_ctx is not None:
        ctx = {'rst': rst_helpers}
        ctx.update(page_ctx)
        environment = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        src = source[0]
        rendered = environment.from_string(src).render(**ctx)
        source[0] = rendered
        Path(utils.docs_dir / '_build' / f'{docname}.rst.out').write_text(rendered)

def setup(app):
    if False:
        while True:
            i = 10
    app.connect('source-read', rstjinja)
    app.add_config_value('jinja_contexts', {}, rebuild='')