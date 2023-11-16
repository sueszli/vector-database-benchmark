"""Doorway module extension for Sphinx.

This extension modifies the way the top-level "falcon" doorway module
is documented.
"""

def _on_process_docstring(app, what, name, obj, options, lines):
    if False:
        i = 10
        return i + 15
    'Process the docstring for a given python object.'
    if what == 'module' and name == 'falcon':
        lines[:] = []

def setup(app):
    if False:
        i = 10
        return i + 15
    app.connect('autodoc-process-docstring', _on_process_docstring)
    return {'parallel_read_safe': True}