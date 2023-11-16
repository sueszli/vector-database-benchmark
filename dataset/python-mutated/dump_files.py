"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.debug as debug
import volatility.plugins.mac.common as common
import volatility.plugins.mac.list_files as mac_list_files

class mac_dump_file(common.AbstractMacCommand):
    """ Dumps a specified file """

    def __init__(self, config, *args, **kwargs):
        if False:
            return 10
        common.AbstractMacCommand.__init__(self, config, *args, **kwargs)
        self._config.add_option('FILE-OFFSET', short_option='q', default=None, help='Virtual address of vnode structure from mac_list_files', action='store', type='int')
        self._config.add_option('OUTFILE', short_option='O', default=None, help='output file path', action='store', type='str')

    def calculate(self):
        if False:
            print('Hello World!')
        common.set_plugin_members(self)
        outfile = self._config.outfile
        vnode_off = self._config.FILE_OFFSET
        if not outfile:
            debug.error('You must specify an output file (-O/--outfile)')
        if not vnode_off:
            debug.error('You must specificy a vnode address (-q/--file-offset) from mac_list_files')
        vnode = obj.Object('vnode', offset=vnode_off, vm=self.addr_space)
        wrote = common.write_vnode_to_file(vnode, outfile)
        yield (vnode_off, outfile, wrote)

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        for (vnode_off, outfile, wrote) in data:
            outfd.write('Wrote {0} bytes to {1} from vnode at address {2:x}\n'.format(wrote, outfile, vnode_off))