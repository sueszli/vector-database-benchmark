"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.plugins.mac.pslist as pslist
import volatility.obj as obj
import volatility.plugins.mac.common as common

class mac_pgrp_hash_table(pslist.mac_pslist):
    """ Walks the process group hash table """

    def calculate(self):
        if False:
            print('Hello World!')
        common.set_plugin_members(self)
        pgrphash_addr = self.addr_space.profile.get_symbol('_pgrphash')
        pgrphash = obj.Object('unsigned long', offset=pgrphash_addr, vm=self.addr_space)
        pgrphashtbl_addr = self.addr_space.profile.get_symbol('_pgrphashtbl')
        pgrphashtbl_ptr = obj.Object('Pointer', offset=pgrphashtbl_addr, vm=self.addr_space)
        pgrphash_array = obj.Object('Array', targetType='pgrphashhead', count=pgrphash + 1, vm=self.addr_space, offset=pgrphashtbl_ptr)
        for plist in pgrphash_array:
            pgrp = plist.lh_first
            while pgrp:
                p = pgrp.pg_members.lh_first
                while p:
                    yield p
                    p = p.p_pglist.le_next
                pgrp = pgrp.pg_hash.le_next