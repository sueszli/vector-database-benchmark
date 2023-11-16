"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization:
"""
import volatility.obj as obj
import volatility.plugins.linux.common as linux_common

class linux_iomem(linux_common.AbstractLinuxCommand):
    """Provides output similar to /proc/iomem"""

    def yield_resource(self, io_res, depth=0):
        if False:
            print('Hello World!')
        if not io_res:
            return []
        name = io_res.name.dereference_as('String', length=linux_common.MAX_STRING_LENGTH)
        start = io_res.start
        end = io_res.end
        output = [(depth, name, start, end)]
        output += self.yield_resource(io_res.child, depth + 1)
        output += self.yield_resource(io_res.sibling, depth)
        return output

    def calculate(self):
        if False:
            print('Hello World!')
        linux_common.set_plugin_members(self)
        io_ptr = self.addr_space.profile.get_symbol('iomem_resource')
        io_res = obj.Object('resource', offset=io_ptr, vm=self.addr_space)
        for r in self.yield_resource(io_res.child):
            yield r

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        for output in data:
            (depth, name, start, end) = output
            outfd.write('{0:35s}\t0x{1:<16X}\t0x{2:<16X}\n'.format('  ' * depth + name, start, end))