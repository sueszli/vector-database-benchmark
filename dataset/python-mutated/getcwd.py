"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.pslist as linux_pslist

class linux_getcwd(linux_pslist.linux_pslist):
    """Lists current working directory of each process"""

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        self.table_header(outfd, [('Name', '17'), ('Pid', '8'), ('CWD', '')])
        for task in data:
            self.table_row(outfd, str(task.comm), task.pid, task.getcwd())