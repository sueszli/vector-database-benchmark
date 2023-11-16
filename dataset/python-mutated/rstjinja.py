import re

def render_with_jinja(docname, source):
    if False:
        i = 10
        return i + 15
    if re.search('^\\s*.. jinja$', source[0], re.M):
        return True
    return False

def rstjinja(app, docname, source):
    if False:
        for i in range(10):
            print('nop')
    '\n    Render our pages as a jinja template for fancy templating goodness.\n    '
    if app.builder.format not in ('html', 'latex'):
        return
    if not render_with_jinja(docname, source):
        return
    src = rendered = source[0]
    print(f'rendering {docname} as jinja templates')
    if app.builder.format == 'html':
        rendered = app.builder.templates.render_string(src, app.config.html_context)
    else:
        from sphinx.util.template import BaseRenderer
        renderer = BaseRenderer()
        rendered = renderer.render_string(src, app.config.html_context)
    source[0] = rendered

def setup(app):
    if False:
        for i in range(10):
            print('nop')
    app.connect('source-read', rstjinja)