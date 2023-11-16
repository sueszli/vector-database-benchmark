"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.plugins.linux.pslist as linux_pslist

class linux_dynamic_env(linux_pslist.linux_pslist):
    """Recover a process' dynamic environment variables"""

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        self.table_header(outfd, [('Pid', '8'), ('Name', '20'), ('Vars', '')])
        for task in data:
            varstr = ''
            for (key, val) in task.bash_environment():
                varstr = varstr + '%s=%s ' % (key, val)
            self.table_row(outfd, task.pid, task.comm, varstr)