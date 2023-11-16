from volatility import renderers
import volatility.plugins.common as common
import volatility.cache as cache
from volatility.renderers.basic import Address, Hex
import volatility.win32 as win32
import volatility.utils as utils

class Modules(common.AbstractWindowsCommand):
    """Print list of loaded modules"""

    def __init__(self, config, *args, **kwargs):
        if False:
            print('Hello World!')
        common.AbstractWindowsCommand.__init__(self, config, *args, **kwargs)
        config.add_option('PHYSICAL-OFFSET', short_option='P', default=False, cache_invalidator=False, help='Physical Offset', action='store_true')

    def generator(self, data):
        if False:
            while True:
                i = 10
        for module in data:
            if not self._config.PHYSICAL_OFFSET:
                offset = module.obj_offset
            else:
                offset = module.obj_vm.vtop(module.obj_offset)
            yield (0, [Address(offset), str(module.BaseDllName or ''), Address(module.DllBase), Hex(module.SizeOfImage), str(module.FullDllName or '')])

    def unified_output(self, data):
        if False:
            i = 10
            return i + 15
        offsettype = '(V)' if not self._config.PHYSICAL_OFFSET else '(P)'
        tg = renderers.TreeGrid([('Offset{0}'.format(offsettype), Address), ('Name', str), ('Base', Address), ('Size', Hex), ('File', str)], self.generator(data))
        return tg

    def render_text(self, outfd, data):
        if False:
            return 10
        offsettype = '(V)' if not self._config.PHYSICAL_OFFSET else '(P)'
        self.table_header(outfd, [('Offset{0}'.format(offsettype), '[addrpad]'), ('Name', '20'), ('Base', '[addrpad]'), ('Size', '[addr]'), ('File', '')])
        for module in data:
            if not self._config.PHYSICAL_OFFSET:
                offset = module.obj_offset
            else:
                offset = module.obj_vm.vtop(module.obj_offset)
            self.table_row(outfd, offset, str(module.BaseDllName or ''), module.DllBase, module.SizeOfImage, str(module.FullDllName or ''))

    @cache.CacheDecorator('tests/lsmod')
    def calculate(self):
        if False:
            while True:
                i = 10
        addr_space = utils.load_as(self._config)
        result = win32.modules.lsmod(addr_space)
        return result

class UnloadedModules(common.AbstractWindowsCommand):
    """Print list of unloaded modules"""

    def unified_output(self, data):
        if False:
            print('Hello World!')

        def generator(data):
            if False:
                return 10
            for drv in data:
                yield (0, [str(drv.Name), Address(drv.StartAddress), Address(drv.EndAddress), str(drv.CurrentTime)])
        return renderers.TreeGrid([('Name', str), ('StartAddress', Address), ('EndAddress', Address), ('Time', str)], generator(data))

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        self.table_header(outfd, [('Name', '20'), ('StartAddress', '[addrpad]'), ('EndAddress', '[addrpad]'), ('Time', '')])
        for drv in data:
            self.table_row(outfd, drv.Name, drv.StartAddress, drv.EndAddress, drv.CurrentTime)

    def calculate(self):
        if False:
            for i in range(10):
                print('nop')
        addr_space = utils.load_as(self._config)
        kdbg = win32.tasks.get_kdbg(addr_space)
        for drv in kdbg.MmUnloadedDrivers.dereference().dereference():
            yield drv