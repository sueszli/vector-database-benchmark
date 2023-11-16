"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.plugins.mac.common as common
import volatility.debug as debug

class mac_find_aslr_shift(common.AbstractMacCommand):
    """ Find the ASLR shift value for 10.8+ images """

    def calculate(self):
        if False:
            for i in range(10):
                print('nop')
        common.set_plugin_members(self)
        yield self.profile.shift_address

    def render_text(self, outfd, data):
        if False:
            return 10
        self.table_header(outfd, [('Shift Value', '#018x')])
        for shift_address in data:
            if shift_address == 0:
                debug.error('Shift addresses are only required on 10.8+ images')
            else:
                self.table_row(outfd, shift_address)