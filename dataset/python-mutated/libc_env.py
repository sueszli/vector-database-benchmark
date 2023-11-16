"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import struct
from operator import attrgetter
import volatility.obj as obj
import volatility.debug as debug
import volatility.addrspace as addrspace
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.pslist as linux_pslist

class linux_bash_env(linux_pslist.linux_pslist):
    """Recover a process' dynamic environment variables"""

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        self.table_header(outfd, [('Pid', '8'), ('Name', '20'), ('Vars', '')])
        for task in data:
            varstr = ''
            for (key, val) in task.bash_environment():
                varstr = varstr + '%s=%s ' % (key, val)
            self.table_row(outfd, task.pid, task.comm, varstr)