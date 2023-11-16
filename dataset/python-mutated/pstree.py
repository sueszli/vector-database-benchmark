"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
"""
import volatility.plugins.linux.pslist as linux_pslist
from volatility.renderers.basic import Address
from volatility.renderers import TreeGrid
from collections import OrderedDict

class linux_pstree(linux_pslist.linux_pslist):
    """Shows the parent/child relationship between processes"""

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.procs = {}
        linux_pslist.linux_pslist.__init__(self, *args, **kwargs)

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        return TreeGrid([('Offset', Address), ('Name', str), ('Level', str), ('Pid', int), ('Ppid', int), ('Uid', int), ('Gid', int), ('Euid', int)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        self.procs = OrderedDict()
        for task in data:
            self.recurse_task(task, 0, 0, self.procs)
        for (offset, name, level, pid, ppid, uid, euid, gid) in self.procs.values():
            if offset:
                yield (0, [Address(offset), str(name), str(level), int(pid), int(ppid), int(uid), int(gid), int(euid)])

    def recurse_task(self, task, ppid, level, procs):
        if False:
            i = 10
            return i + 15
        '\n        Fill a dictionnary with all the children of a given task(including itself)\n        :param task: task that we want to get the children from\n        :param ppid: pid of the parent task\n        :param level: depth from the root task\n        :param procs: dictionnary that we fill\n        '
        if not procs.has_key(task.pid.v()):
            if task.mm:
                proc_name = task.comm
            else:
                proc_name = '[' + task.comm + ']'
            procs[task.pid.v()] = (task.obj_offset, proc_name, '.' * level + proc_name, task.pid, ppid, task.uid, task.euid, task.gid)
            for child in task.children.list_of_type('task_struct', 'sibling'):
                self.recurse_task(child, task.pid, level + 1, procs)

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        self.procs = OrderedDict()
        outfd.write('{0:20s} {1:15s} {2:15s}\n'.format('Name', 'Pid', 'Uid'))
        for task in data:
            self.recurse_task(task, 0, 0, self.procs)
        for (offset, _, proc_name, pid, _, uid, _, _) in self.procs.values():
            if offset:
                outfd.write('{0:20s} {1:15s} {2:15s}\n'.format(proc_name, str(pid), str(uid or '')))