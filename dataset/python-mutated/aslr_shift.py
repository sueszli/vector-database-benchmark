"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.utils as utils
import volatility.plugins.linux.common as common

class linux_aslr_shift(common.AbstractLinuxCommand):
    """Automatically detect the Linux ASLR shift"""

    def calculate(self):
        if False:
            while True:
                i = 10
        aspace = utils.load_as(self._config)
        yield (aspace.profile.virtual_shift, aspace.profile.physical_shift)

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        self.table_header(outfd, [('Virtual Shift Address', '[addrpad]'), ('Physical Shift Address', '[addrpad]')])
        for (v, p) in data:
            self.table_row(outfd, v, p)