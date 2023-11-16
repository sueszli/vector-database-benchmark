import types
from .utils import ExactWeakKeyDictionary

class CodeContextDict:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.code_context = ExactWeakKeyDictionary()

    def has_context(self, code: types.CodeType):
        if False:
            return 10
        return code in self.code_context

    def get_context(self, code: types.CodeType):
        if False:
            return 10
        ctx = self.code_context.get(code)
        if ctx is None:
            ctx = {}
            self.code_context[code] = ctx
        return ctx

    def pop_context(self, code: types.CodeType):
        if False:
            i = 10
            return i + 15
        ctx = self.get_context(code)
        self.code_context._remove_id(id(code))
        return ctx

    def clear(self):
        if False:
            i = 10
            return i + 15
        self.code_context.clear()
code_context = CodeContextDict()