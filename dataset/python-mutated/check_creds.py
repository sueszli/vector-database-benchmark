"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.debug as debug
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.pslist as linux_pslist
from volatility.renderers import TreeGrid

class linux_check_creds(linux_pslist.linux_pslist):
    """Checks if any processes are sharing credential structures"""

    def calculate(self):
        if False:
            for i in range(10):
                print('nop')
        linux_common.set_plugin_members(self)
        if not self.profile.obj_has_member('task_struct', 'cred'):
            debug.error('This command is not supported in this profile.')
        creds = {}
        tasks = linux_pslist.linux_pslist.calculate(self)
        for task in tasks:
            cred_addr = task.cred.v()
            if not cred_addr in creds:
                creds[cred_addr] = []
            creds[cred_addr].append(task.pid)
        yield creds

    def unified_output(self, data):
        if False:
            return 10
        return TreeGrid([('PIDs', str)], self.generator(data))

    def generator(self, data):
        if False:
            for i in range(10):
                print('nop')
        for htable in data:
            for (addr, pids) in htable.items():
                if len(pids) > 1:
                    pid_str = ''
                    for pid in pids:
                        pid_str = pid_str + '{0:d}, '.format(pid)
                    pid_str = pid_str[:-2]
                    yield (0, [str(pid_str)])

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        self.table_header(outfd, [('PIDs', '8')])
        for htable in data:
            for (addr, pids) in htable.items():
                if len(pids) > 1:
                    pid_str = ''
                    for pid in pids:
                        pid_str = pid_str + '{0:d}, '.format(pid)
                    pid_str = pid_str[:-2]
                    self.table_row(outfd, pid_str)