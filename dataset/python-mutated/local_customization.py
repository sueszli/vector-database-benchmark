from docutils.parsers.rst import directives as directives
from sphinx import addnodes
from sphinx.domains.python import PyClasslike
from sphinx.ext.autodoc import ClassLevelDocumenter as ClassLevelDocumenter, FunctionDocumenter as FunctionDocumenter, MethodDocumenter as MethodDocumenter, Options as Options
'\n\n.. interface:: The nursery interface\n\n   .. attribute:: blahblah\n\n'

class Interface(PyClasslike):

    def handle_signature(self, sig, signode):
        if False:
            for i in range(10):
                print('nop')
        signode += addnodes.desc_name(sig, sig)
        return (sig, '')

    def get_index_text(self, modname, name_cls):
        if False:
            i = 10
            return i + 15
        return f'{name_cls[0]} (interface in {modname})'

def setup(app):
    if False:
        return 10
    app.add_directive_to_domain('py', 'interface', Interface)