"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.pslist as linux_pslist
import volatility.plugins.linux.proc_maps as linux_proc_maps

class linux_proc_maps_rb(linux_proc_maps.linux_proc_maps):
    """Gathers process maps for linux through the mappings red-black tree"""

    def calculate(self):
        if False:
            while True:
                i = 10
        linux_common.set_plugin_members(self)
        tasks = linux_pslist.linux_pslist.calculate(self)
        for task in tasks:
            if task.mm:
                for vma in task.get_proc_maps_rb():
                    yield (task, vma)