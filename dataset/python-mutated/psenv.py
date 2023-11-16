"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.plugins.linux.pslist as linux_pslist
from volatility.renderers import TreeGrid

class linux_psenv(linux_pslist.linux_pslist):
    """Gathers processes along with their static environment variables"""

    def unified_output(self, data):
        if False:
            i = 10
            return i + 15
        return TreeGrid([('Name', str), ('Pid', int), ('Environment', str)], self.generator(data))

    def generator(self, data):
        if False:
            return 10
        for task in data:
            yield (0, [str(task.comm), int(task.pid), str(task.get_environment())])

    def render_text(self, outfd, data):
        if False:
            return 10
        outfd.write('{0:6s} {1:6s} {2:12s}\n'.format('Name', 'Pid', 'Environment'))
        for task in data:
            outfd.write('{0:17s} {1:6s} {2:s}\n'.format(str(task.comm), str(task.pid), task.get_environment()))