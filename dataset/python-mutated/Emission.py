""" Emission of source code.

Code generation is driven via "emit", which is to receive lines of code and
this is to collect them, providing the emit implementation. Sometimes nested
use of these will occur.

"""
import contextlib
from .Indentation import indented

class SourceCodeCollector(object):

    def __init__(self):
        if False:
            return 10
        self.codes = []

    def __call__(self, code):
        if False:
            i = 10
            return i + 15
        self.emit(code)

    def emit(self, code):
        if False:
            i = 10
            return i + 15
        for line in code.split('\n'):
            self.codes.append(line)

    def emitTo(self, emit, level):
        if False:
            print('Hello World!')
        for code in self.codes:
            emit(indented(code, level))
        self.codes = None

@contextlib.contextmanager
def withSubCollector(emit, context):
    if False:
        i = 10
        return i + 15
    context.pushCleanupScope()
    with context.variable_storage.withLocalStorage():
        sub_emit = SourceCodeCollector()
        yield sub_emit
        local_declarations = context.variable_storage.makeCLocalDeclarations()
        if local_declarations:
            emit('{')
            for local_declaration in local_declarations:
                emit(indented(local_declaration))
            sub_emit.emitTo(emit, level=1)
            emit('}')
        else:
            sub_emit.emitTo(emit, level=0)
        context.popCleanupScope()