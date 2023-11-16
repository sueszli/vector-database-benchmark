from bisect import bisect_right
import volatility.plugins.mac.pstasks as mac_tasks
import volatility.plugins.strings as strings
import volatility.plugins.mac.common as mac_common
import volatility.plugins.mac.lsmod as mac_lsmod

class mac_strings(strings.Strings, mac_common.AbstractMacCommand):
    """Match physical offsets to virtual addresses (may take a while, VERY verbose)"""

    @staticmethod
    def is_valid_profile(profile):
        if False:
            for i in range(10):
                print('nop')
        return profile.metadata.get('os', 'Unknown').lower() == 'mac'

    def get_processes(self, addr_space):
        if False:
            for i in range(10):
                print('nop')
        'Enumerate processes based on user options.\n\n        :param      addr_space | <addrspace.AbstractVirtualAddressSpace>\n\n        :returns    <list> \n        '
        tasks = mac_tasks.mac_tasks(self._config).calculate()
        try:
            if self._config.PID is not None:
                pidlist = [int(p) for p in self._config.PID.split(',')]
                tasks = [t for t in tasks if int(t.p_pid) in pidlist]
        except (ValueError, TypeError):
            debug.error('Invalid PID {0}'.format(self._config.PID))
        return tasks

    @classmethod
    def get_modules(cls, addr_space):
        if False:
            i = 10
            return i + 15
        'Enumerate the kernel modules. \n\n        :param      addr_space | <addrspace.AbstractVirtualAddressSpace>\n        \n        :returns    <tuple>\n        '
        mask = addr_space.address_mask
        config = addr_space.get_config()
        modules = mac_lsmod.mac_lsmod(config).calculate()
        mods = dict(((mask(mod.address), mod) for mod in modules))
        mod_addrs = sorted(mods.keys())
        return (mods, mod_addrs)

    @classmethod
    def find_module(cls, modlist, mod_addrs, addr_space, vpage):
        if False:
            print('Hello World!')
        'Determine which module owns a virtual page. \n\n        :param      modlist     | <list>\n                    mod_addrs   | <list>\n                    addr_space  | <addrspace.AbstractVirtualAddressSpace>\n                    vpage       | <int> \n        \n        :returns    <module> || None\n        '
        pos = bisect_right(mod_addrs, vpage) - 1
        if pos == -1:
            return None
        mod = modlist[mod_addrs[pos]]
        compare = mod.obj_vm.address_compare
        if compare(vpage, mod.address) != -1 and compare(vpage, mod.address + mod.m('size')) == -1:
            return mod
        else:
            return None

    @classmethod
    def get_module_name(cls, module):
        if False:
            while True:
                i = 10
        'Get the name of a kernel module.\n\n        :param      module      | <module>\n        \n        :returns    <str>\n        '
        return str(module.m('name'))

    @classmethod
    def get_task_pid(cls, task):
        if False:
            for i in range(10):
                print('nop')
        'Get the PID of a process. \n\n        :param      task   | <task>\n        \n        :returns    <int>\n        '
        return task.p_pid