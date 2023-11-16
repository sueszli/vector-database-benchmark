from jedi.inference.value import ModuleValue
from jedi.inference.context import ModuleContext

class DocstringModule(ModuleValue):

    def __init__(self, in_module_context, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self._in_module_context = in_module_context

    def _as_context(self):
        if False:
            while True:
                i = 10
        return DocstringModuleContext(self, self._in_module_context)

class DocstringModuleContext(ModuleContext):

    def __init__(self, module_value, in_module_context):
        if False:
            i = 10
            return i + 15
        super().__init__(module_value)
        self._in_module_context = in_module_context

    def get_filters(self, origin_scope=None, until_position=None):
        if False:
            for i in range(10):
                print('nop')
        yield from super().get_filters(until_position=until_position)
        yield from self._in_module_context.get_filters()