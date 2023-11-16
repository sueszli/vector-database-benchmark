"""
Escape the `body` part of .chm source file to 7-bit ASCII, to fix visual
effect on some MBCS Windows systems.

https://bugs.python.org/issue32174
"""
import pathlib
import re
from html.entities import codepoint2name
from sphinx.util.logging import getLogger

def _process(string):
    if False:
        return 10

    def escape(matchobj):
        if False:
            for i in range(10):
                print('nop')
        codepoint = ord(matchobj.group(0))
        name = codepoint2name.get(codepoint)
        if name is None:
            return '&#%d;' % codepoint
        else:
            return '&%s;' % name
    return re.sub('[^\\x00-\\x7F]', escape, string)

def escape_for_chm(app, pagename, templatename, context, doctree):
    if False:
        while True:
            i = 10
    if getattr(app.builder, 'name', '') != 'htmlhelp':
        return
    body = context.get('body')
    if body is not None:
        context['body'] = _process(body)

def fixup_keywords(app, exception):
    if False:
        print('Hello World!')
    if getattr(app.builder, 'name', '') != 'htmlhelp' or exception:
        return
    getLogger(__name__).info('fixing HTML escapes in keywords file...')
    outdir = pathlib.Path(app.builder.outdir)
    outname = app.builder.config.htmlhelp_basename
    with open(outdir / (outname + '.hhk'), 'rb') as f:
        index = f.read()
    with open(outdir / (outname + '.hhk'), 'wb') as f:
        f.write(index.replace(b'&#x27;', b'&#39;'))

def setup(app):
    if False:
        while True:
            i = 10
    app.connect('html-page-context', escape_for_chm)
    app.connect('build-finished', fixup_keywords)
    return {'version': '1.0', 'parallel_read_safe': True}