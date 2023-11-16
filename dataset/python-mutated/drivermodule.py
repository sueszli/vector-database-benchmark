import volatility.utils as utils
import volatility.obj as obj
import volatility.debug as debug
import volatility.plugins.common as common
import volatility.plugins.malware.devicetree as dtree
import volatility.win32.modules as modules
import volatility.win32.tasks as tasks
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class drivermodule(common.AbstractWindowsCommand):
    """Associate driver objects to kernel modules"""

    def __init__(self, config, *args, **kwargs):
        if False:
            print('Hello World!')
        common.AbstractWindowsCommand.__init__(self, config, *args, **kwargs)
        config.add_option('ADDR', short_option='a', default=None, help='Show info on module at or containing this (base) address', action='store', type='int')

    def calculate(self):
        if False:
            print('Hello World!')
        addr_space = utils.load_as(self._config)
        modlist = list(modules.lsmod(addr_space))
        mods = dict(((addr_space.address_mask(mod.DllBase), mod) for mod in modlist))
        mod_addrs = sorted(mods.keys())
        drivers = dtree.DriverIrp(self._config).calculate()
        driver_name = 'UNKNOWN'
        service_key = 'UNKNOWN'
        driver_name3 = 'UNKNOWN'
        module_name = 'UNKNOWN'
        if self._config.ADDR:
            find_address = self._config.ADDR
            module_name = tasks.find_module(mods, mod_addrs, mods.values()[0].obj_vm.address_mask(find_address))
            if module_name:
                module_name = module_name.BaseDllName or module_name.FullDllName
            for driver in drivers:
                if driver.DriverStart <= find_address < driver.DriverStart + driver.DriverSize:
                    header = driver.get_object_header()
                    driver_name = header.NameInfo.Name
                    driver_name = str(driver.get_object_header().NameInfo.Name or '')
                    service_key = str(driver.DriverExtension.ServiceKeyName or '')
                    driver_name3 = str(driver.DriverName or '')
                    break
            yield (module_name, driver_name, service_key, driver_name3)
        else:
            for driver in drivers:
                driver_name = str(driver.get_object_header().NameInfo.Name or '')
                service_key = str(driver.DriverExtension.ServiceKeyName or '')
                driver_name3 = str(driver.DriverName or '')
                owning_module = tasks.find_module(mods, mod_addrs, mods.values()[0].obj_vm.address_mask(driver.DriverStart))
                module_name = 'UNKNOWN'
                if owning_module:
                    module_name = owning_module.BaseDllName or owning_module.FullDllName
                yield (module_name, driver_name, service_key, driver_name3)

    def generator(self, data):
        if False:
            while True:
                i = 10
        for (module_name, driver_name, service_key, driver_name3) in data:
            yield (0, [str(module_name), str(driver_name), str(service_key), str(driver_name3)])

    def unified_output(self, data):
        if False:
            return 10
        return TreeGrid([('Module', str), ('Driver', str), ('Alt. Name', str), ('Service Key', str)], self.generator(data))

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        self.table_header(outfd, [('Module', '36'), ('Driver', '24'), ('Alt. Name', '24'), ('Service Key', '')])
        for (module_name, driver_name, service_key, driver_name3) in data:
            self.table_row(outfd, module_name, driver_name, service_key, driver_name3)