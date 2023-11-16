import volatility.plugins.crashinfo as crashinfo
import volatility.plugins.linux.common as linux_common

class LiMEInfo(linux_common.AbstractLinuxCommand):
    """Dump Lime file format information"""
    target_as = ['LimeAddressSpace']

    def calculate(self):
        if False:
            return 10
        'Determines the address space'
        linux_common.set_plugin_members(self)
        result = None
        adrs = self.addr_space
        while adrs:
            if adrs.__class__.__name__ in self.target_as:
                result = adrs
            adrs = adrs.base
        if result is None:
            debug.error('Memory Image could not be identified as {0}'.format(self.target_as))
        return result

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        self.table_header(outfd, [('Memory Start', '[addrpad]'), ('Memory End', '[addrpad]'), ('Size', '[addrpad]')])
        for seg in data.runs:
            self.table_row(outfd, seg[0], seg[0] + seg[2] - 1, seg[2])