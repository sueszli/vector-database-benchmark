"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization:
"""
import volatility.obj as obj
import volatility.debug as debug
import volatility.plugins.linux.flags as linux_flags
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.pslist as linux_pslist

class linux_banner(linux_common.AbstractLinuxCommand):
    """ Prints the Linux banner information """

    def calculate(self):
        if False:
            return 10
        linux_common.set_plugin_members(self)
        banner_addr = self.addr_space.profile.get_symbol('linux_banner')
        if banner_addr:
            banner = obj.Object('String', offset=banner_addr, vm=self.addr_space, length=256)
        else:
            debug.error('linux_banner symbol not found. Please report this as a bug on the issue tracker: https://code.google.com/p/volatility/issues/list')
        yield banner.strip()

    def render_text(self, outfd, data):
        if False:
            return 10
        for banner in data:
            outfd.write('{0:s}\n'.format(banner))