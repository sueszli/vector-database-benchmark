""" Generate docs for examples.
"""
import os
import json
from types import ModuleType
import flexx
from flexx import ui, app
from urllib.request import urlopen, Request
from uiexample import create_ui_example
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
EXAMPLES_DIR = os.path.join(os.path.dirname(DOC_DIR), 'flexxamples')
EXAMPLES_DIRS = []
for dname in os.listdir(EXAMPLES_DIR):
    dirname = os.path.join(EXAMPLES_DIR, dname)
    if os.path.isfile(os.path.join(dirname, '__init__.py')):
        EXAMPLES_DIRS.append((dname, dirname))
created_files = []

def get_notebook_list():
    if False:
        while True:
            i = 10
    url = 'https://api.github.com/repos/flexxui/flexx-notebooks/contents'
    print('downloading %s ... ' % url, end='')
    req = Request(url, headers={'User-Agent': 'flexx/%s' % flexx.__version__})
    s = json.loads(urlopen(req, timeout=5.0).read().decode())
    print('done')
    filenames = []
    for file in s:
        if file['name'].endswith('ipynb'):
            filenames.append(file['name'])
    return filenames
EXAMPLES_TEXT = '\nThis page provides a list of examples. Some demonstate a specific application,\nwhile others show a tool/technique that is generically useful. In the latter case\nyou can import the widget using e.g. ``from flexxamples.howtos.editor_cm import CodeEditor``.\n\nNote that most examples are written in such a way that they work in the\nbrowser (by subclassing :class:`Widget <flexx.ui.Widget>`). If you are creating a desktop\napplication, you probably want to use :class:`PyWidget <flexx.ui.PyWidget>` to create your\nhigh-level widgets instead.\n'

def main():
    if False:
        while True:
            i = 10
    output_dir = os.path.join(DOC_DIR, 'examples')
    examples = {}
    for (sub, dirname) in EXAMPLES_DIRS:
        examples[sub] = {}
        for fname in os.listdir(dirname):
            filename = os.path.join(dirname, fname)
            if os.path.isfile(filename) and fname.endswith('.py') and (fname[0] != '_'):
                code = open(filename, 'rt', encoding='utf-8').read()
                text = ':orphan:\n\n'
                text += '.. _%s:\n\n' % fname
                text += '%s\n%s\n\n' % (fname, '=' * len(fname))
                if code.startswith('# doc-export:'):
                    code = code.split('\n', 1)[1].lstrip()
                    html = create_ui_example(filename, '..', source=filename)
                    text += '.. raw:: html\n\n    ' + html + '\n\n'
                text += '.. code-block:: py\n    :linenos:\n\n'
                text += '\n    ' + code.replace('\n', '\n    ').rstrip() + '\n'
                examples[sub][fname] = text
        if not examples[sub]:
            del examples[sub]
        for name in examples[sub]:
            filename = os.path.join(output_dir, name[:-3] + '_src.rst')
            created_files.append(filename)
            open(filename, 'wt', encoding='utf-8').write(examples[sub][name])
    docs = 'Examples'
    docs += '\n%s\n\n' % (len(docs) * '=')
    docs += EXAMPLES_TEXT + '\n\n'
    for (sub, _) in EXAMPLES_DIRS:
        section = sub.capitalize()
        docs += '\n%s\n%s\n\n' % (section, len(section) * '-')
        for name in sorted(examples[sub]):
            docs += '* :ref:`%s`\n' % name
    filename = os.path.join(DOC_DIR, 'examples', 'index.rst')
    created_files.append(filename)
    open(filename, 'wt', encoding='utf-8').write(docs)
    print('  generated %i examples' % sum([len(x) for x in examples.values()]))

def clean():
    if False:
        while True:
            i = 10
    while created_files:
        filename = created_files.pop()
        if os.path.isfile(filename):
            os.remove(filename)
        elif os.path.isdir(filename) and (not os.listdir(filename)):
            os.rmdir(filename)