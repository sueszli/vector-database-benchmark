"""Private args module extension for Sphinx.

This extension excludes underscore-prefixed args and kwargs from the function
signature.
"""

def _on_process_signature(app, what, name, obj, options, signature, return_annotation):
    if False:
        i = 10
        return i + 15
    if what in ('function', 'method') and signature and ('_' in signature):
        filtered = []
        for token in signature[1:-1].split(','):
            token = token.strip()
            if not token.startswith('_'):
                filtered.append(token)
        signature = f"({', '.join(filtered)})"
    return (signature, return_annotation)

def setup(app):
    if False:
        while True:
            i = 10
    app.connect('autodoc-process-signature', _on_process_signature)
    return {'parallel_read_safe': True}