"""
    :codeauthor: Pedro Algarvio (pedro@algarvio.me)


    saltautodoc.py
    ~~~~~~~~~~~~~~

    Properly handle ``__func_alias__``
"""
from sphinx.ext.autodoc import FunctionDocumenter

class SaltFunctionDocumenter(FunctionDocumenter):
    """
    Simple override of sphinx.ext.autodoc.FunctionDocumenter to properly render
    salt's aliased function names.
    """

    def format_name(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Format the function name\n        '
        if not hasattr(self.module, '__func_alias__'):
            return super(FunctionDocumenter, self).format_name()
        if not self.objpath:
            return super(FunctionDocumenter, self).format_name()
        if len(self.objpath) > 1:
            return super(FunctionDocumenter, self).format_name()
        return self.module.__func_alias__.get(self.objpath[0], self.objpath[0])

def setup(app):
    if False:
        for i in range(10):
            print('nop')

    def add_documenter(app, env, docnames):
        if False:
            print('Hello World!')
        app.add_autodocumenter(SaltFunctionDocumenter)
    app.connect('env-before-read-docs', add_documenter)