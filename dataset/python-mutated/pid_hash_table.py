"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.plugins.mac.pslist as pslist
import volatility.obj as obj
import volatility.plugins.mac.common as common

class mac_pid_hash_table(pslist.mac_pslist):
    """ Walks the pid hash table """

    def calculate(self):
        if False:
            return 10
        common.set_plugin_members(self)
        pidhash_addr = self.addr_space.profile.get_symbol('_pidhash')
        pidhash = obj.Object('unsigned long', offset=pidhash_addr, vm=self.addr_space)
        pidhashtbl_addr = self.addr_space.profile.get_symbol('_pidhashtbl')
        pidhashtbl_ptr = obj.Object('Pointer', offset=pidhashtbl_addr, vm=self.addr_space)
        pidhash_array = obj.Object('Array', targetType='pidhashhead', count=pidhash + 1, vm=self.addr_space, offset=pidhashtbl_ptr)
        for plist in pidhash_array:
            p = plist.lh_first.dereference()
            while p:
                yield p
                p = p.p_hash.le_next.dereference()