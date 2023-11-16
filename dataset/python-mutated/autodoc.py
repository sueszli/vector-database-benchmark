from faker.sphinx.docstring import ProviderMethodDocstring
from faker.sphinx.documentor import write_provider_docs

def _create_source_files(app):
    if False:
        return 10
    write_provider_docs()

def _process_docstring(app, what, name, obj, options, lines):
    if False:
        i = 10
        return i + 15
    docstring = ProviderMethodDocstring(app, what, name, obj, options, lines)
    if not docstring.skipped:
        lines[:] = docstring.lines[:]

def setup(app):
    if False:
        i = 10
        return i + 15
    app.setup_extension('sphinx.ext.autodoc')
    app.connect('builder-inited', _create_source_files)
    app.connect('autodoc-process-docstring', _process_docstring)