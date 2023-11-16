"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.mac.pslist as pslist
import volatility.plugins.mac.common as common

class mac_tasks(pslist.mac_pslist):
    """ List Active Tasks """

    def __init__(self, config, *args, **kwargs):
        if False:
            return 10
        pslist.mac_pslist.__init__(self, config, *args, **kwargs)

    def allprocs(self):
        if False:
            print('Hello World!')
        common.set_plugin_members(self)
        tasksaddr = self.addr_space.profile.get_symbol('_tasks')
        queue_entry = obj.Object('queue_entry', offset=tasksaddr, vm=self.addr_space)
        seen = {tasksaddr: 1}
        for task in queue_entry.walk_list(list_head=tasksaddr):
            if task.obj_offset not in seen:
                seen[task.obj_offset] = 0
                if task.bsd_info:
                    proc = task.bsd_info.dereference_as('proc')
                    yield proc
            else:
                if seen[task.obj_offset] > 3:
                    break
                seen[task.obj_offset] = seen[task.obj_offset] + 1