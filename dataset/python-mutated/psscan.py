"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import struct
import volatility.obj as obj
import volatility.utils as utils
import volatility.poolscan as poolscan
import volatility.debug as debug
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.pslist as pslist
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class linux_psscan(pslist.linux_pslist):
    """ Scan physical memory for processes """

    def __init__(self, config, *args, **kwargs):
        if False:
            print('Hello World!')
        linux_common.AbstractLinuxCommand.__init__(self, config, *args, **kwargs)
        self.wants_physical = True

    def calculate(self):
        if False:
            i = 10
            return i + 15
        linux_common.set_plugin_members(self)
        phys_addr_space = utils.load_as(self._config, astype='physical')
        if phys_addr_space.profile.metadata.get('memory_model', '32bit') == '32bit':
            fmt = '<I'
        else:
            fmt = '<Q'
        needles = []
        for sym in phys_addr_space.profile.get_all_symbol_names('kernel'):
            if sym.find('_sched_class') != -1:
                addr = phys_addr_space.profile.get_symbol(sym)
                needles.append(struct.pack(fmt, addr))
        if len(needles) == 0:
            debug.warning('Unable to scan for processes. Please file a bug report.')
        else:
            back_offset = phys_addr_space.profile.get_obj_offset('task_struct', 'sched_class')
            scanner = poolscan.MultiPoolScanner(needles)
            for (_, offset) in scanner.scan(phys_addr_space):
                ptask = obj.Object('task_struct', offset=offset - back_offset, vm=phys_addr_space)
                if not ptask.exit_state.v() in [0, 16, 32, 16 | 32]:
                    continue
                if not 0 < ptask.pid < 66000:
                    continue
                yield ptask