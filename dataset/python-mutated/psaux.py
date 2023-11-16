"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.plugins.linux.pslist as linux_pslist
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class linux_psaux(linux_pslist.linux_pslist):
    """Gathers processes along with full command line and start time"""

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        return TreeGrid([('Arguments', str), ('Pid', int), ('Uid', int), ('Gid', int)], self.generator(data))

    def generator(self, data):
        if False:
            return 10
        for task in data:
            yield (0, [str(task.get_commandline()), int(task.pid), int(task.uid), int(task.gid)])

    def render_text(self, outfd, data):
        if False:
            return 10
        outfd.write('{1:6s} {2:6s} {3:6s} {0:64s}\n'.format('Arguments', 'Pid', 'Uid', 'Gid'))
        for task in data:
            outfd.write('{1:6s} {2:6s} {3:6s} {0:64s}\n'.format(task.get_commandline(), str(task.pid), str(task.uid), str(task.gid)))