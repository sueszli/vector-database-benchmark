"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.mac.pstasks as pstasks
import volatility.plugins.mac.common as common
from volatility.renderers import TreeGrid

class mac_keychaindump(pstasks.mac_tasks):
    """ Recovers possbile keychain keys. Use chainbreaker to open related keychain files """

    def calculate(self):
        if False:
            while True:
                i = 10
        common.set_plugin_members(self)
        procs = pstasks.mac_tasks.calculate(self)
        if self.addr_space.profile.metadata.get('memory_model', '32bit') == '32bit':
            ptr_sz = 4
        else:
            ptr_sz = 8
        for proc in procs:
            if str(proc.p_comm) != 'securityd':
                continue
            proc_as = proc.get_process_address_space()
            for map in proc.get_proc_maps():
                if not (map.start > 139637976727552 and map.end < 140733193388032 and (map.end - map.start == 1048576)):
                    continue
                for address in range(map.start, map.end, ptr_sz):
                    signature = obj.Object('unsigned int', offset=address, vm=proc_as)
                    if not signature or signature != 24:
                        continue
                    key_buf_ptr = obj.Object('unsigned long', offset=address + ptr_sz, vm=proc_as)
                    if map.start <= key_buf_ptr < map.end:
                        yield (proc_as, key_buf_ptr)

    def unified_output(self, data):
        if False:
            i = 10
            return i + 15
        return TreeGrid([('Key', str)], self.generator(data))

    def generator(self, data):
        if False:
            while True:
                i = 10
        for (proc_as, key_buf_ptr) in data:
            key_buf = proc_as.read(key_buf_ptr, 24)
            if not key_buf:
                continue
            key = ''.join(('%02X' % ord(k) for k in key_buf))
            yield (0, [str(key)])

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        self.table_header(outfd, [('Key', '')])
        for (proc_as, key_buf_ptr) in data:
            key_buf = proc_as.read(key_buf_ptr, 24)
            if not key_buf:
                continue
            key = ''.join(('%02X' % ord(k) for k in key_buf))
            self.table_row(outfd, key)