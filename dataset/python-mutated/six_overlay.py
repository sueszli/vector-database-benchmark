"""Implementation of special members of third_party/six."""
from pytype.overlays import metaclass
from pytype.overlays import overlay

class SixOverlay(overlay.Overlay):
    """A custom overlay for the 'six' module."""

    def __init__(self, ctx):
        if False:
            i = 10
            return i + 15
        member_map = {'add_metaclass': metaclass.AddMetaclass.make, 'with_metaclass': metaclass.WithMetaclass.make, 'string_types': overlay.drop_module(build_string_types), 'integer_types': overlay.drop_module(build_integer_types), 'PY2': build_version_bool(2), 'PY3': build_version_bool(3)}
        ast = ctx.loader.import_name('six')
        super().__init__(ctx, 'six', member_map, ast)

def build_version_bool(major):
    if False:
        return 10

    def make(ctx, module):
        if False:
            return 10
        del module
        return ctx.convert.bool_values[ctx.python_version[0] == major]
    return make

def build_string_types(ctx):
    if False:
        return 10
    classes = [ctx.convert.str_type.to_variable(ctx.root_node)]
    return ctx.convert.tuple_to_value(classes)

def build_integer_types(ctx):
    if False:
        while True:
            i = 10
    return ctx.convert.tuple_to_value((ctx.convert.int_type.to_variable(ctx.root_node),))