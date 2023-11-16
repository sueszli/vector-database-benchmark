"""Implementation of special members of sys."""
from pytype.abstract import abstract
from pytype.overlays import overlay

class SysOverlay(overlay.Overlay):
    """A custom overlay for the 'sys' module."""

    def __init__(self, ctx):
        if False:
            i = 10
            return i + 15
        member_map = {'platform': overlay.drop_module(build_platform), 'version_info': overlay.drop_module(build_version_info)}
        ast = ctx.loader.import_name('sys')
        super().__init__(ctx, 'sys', member_map, ast)

class VersionInfo(abstract.Tuple):
    ATTRIBUTES = ('major', 'minor', 'micro', 'releaselevel', 'serial')

    def get_special_attribute(self, node, name, valself):
        if False:
            for i in range(10):
                print('nop')
        try:
            index = self.ATTRIBUTES.index(name)
        except ValueError:
            return None
        return self.pyval[index]

def build_platform(ctx):
    if False:
        i = 10
        return i + 15
    return ctx.convert.constant_to_value(ctx.options.platform)

def build_version_info(ctx):
    if False:
        print('Hello World!')
    'Build sys.version_info.'
    version = []
    for i in ctx.python_version:
        version.append(ctx.convert.constant_to_var(i))
    for t in (int, str, int):
        version.append(ctx.convert.primitive_class_instances[t].to_variable(ctx.root_node))
    return VersionInfo(tuple(version), ctx)