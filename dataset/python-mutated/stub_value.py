from jedi.inference.base_value import ValueWrapper
from jedi.inference.value.module import ModuleValue
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.names import StubName, StubModuleName
from jedi.inference.gradual.typing import TypingModuleFilterWrapper
from jedi.inference.context import ModuleContext

class StubModuleValue(ModuleValue):
    _module_name_class = StubModuleName

    def __init__(self, non_stub_value_set, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.non_stub_value_set = non_stub_value_set

    def is_stub(self):
        if False:
            print('Hello World!')
        return True

    def sub_modules_dict(self):
        if False:
            while True:
                i = 10
        "\n        We have to overwrite this, because it's possible to have stubs that\n        don't have code for all the child modules. At the time of writing this\n        there are for example no stubs for `json.tool`.\n        "
        names = {}
        for value in self.non_stub_value_set:
            try:
                method = value.sub_modules_dict
            except AttributeError:
                pass
            else:
                names.update(method())
        names.update(super().sub_modules_dict())
        return names

    def _get_stub_filters(self, origin_scope):
        if False:
            return 10
        return [StubFilter(parent_context=self.as_context(), origin_scope=origin_scope)] + list(self.iter_star_filters())

    def get_filters(self, origin_scope=None):
        if False:
            print('Hello World!')
        filters = super().get_filters(origin_scope)
        next(filters, None)
        stub_filters = self._get_stub_filters(origin_scope=origin_scope)
        yield from stub_filters
        yield from filters

    def _as_context(self):
        if False:
            print('Hello World!')
        return StubModuleContext(self)

class StubModuleContext(ModuleContext):

    def get_filters(self, until_position=None, origin_scope=None):
        if False:
            return 10
        return super().get_filters(origin_scope=origin_scope)

class TypingModuleWrapper(StubModuleValue):

    def get_filters(self, *args, **kwargs):
        if False:
            print('Hello World!')
        filters = super().get_filters(*args, **kwargs)
        f = next(filters, None)
        assert f is not None
        yield TypingModuleFilterWrapper(f)
        yield from filters

    def _as_context(self):
        if False:
            print('Hello World!')
        return TypingModuleContext(self)

class TypingModuleContext(ModuleContext):

    def get_filters(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        filters = super().get_filters(*args, **kwargs)
        yield TypingModuleFilterWrapper(next(filters, None))
        yield from filters

class StubFilter(ParserTreeFilter):
    name_class = StubName

    def _is_name_reachable(self, name):
        if False:
            print('Hello World!')
        if not super()._is_name_reachable(name):
            return False
        definition = name.get_definition()
        if definition is None:
            return False
        if definition.type in ('import_from', 'import_name'):
            if name.parent.type not in ('import_as_name', 'dotted_as_name'):
                return False
        n = name.value
        if n.startswith('_') and (not (n.startswith('__') and n.endswith('__'))):
            return False
        return True

class VersionInfo(ValueWrapper):
    pass