"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import socket
import volatility.plugins.linux.common as linux_common
import volatility.obj as obj

class a_ent(object):

    def __init__(self, ip, mac, devname):
        if False:
            print('Hello World!')
        self.ip = ip
        self.mac = mac
        self.devname = devname

class linux_arp(linux_common.AbstractLinuxCommand):
    """Print the ARP table"""

    def calculate(self):
        if False:
            while True:
                i = 10
        linux_common.set_plugin_members(self)
        neigh_tables_addr = self.addr_space.profile.get_symbol('neigh_tables')
        hasnext = True
        try:
            self.addr_space.profile.get_obj_offset('neigh_table', 'next')
        except KeyError:
            hasnext = False
        if hasnext == True:
            ntables_ptr = obj.Object('Pointer', offset=neigh_tables_addr, vm=self.addr_space)
            tables = linux_common.walk_internal_list('neigh_table', 'next', ntables_ptr)
        else:
            tables_arr = obj.Object(theType='Array', targetType='Pointer', offset=neigh_tables_addr, vm=self.addr_space, count=4)
            tables = [t.dereference_as('neigh_table') for t in tables_arr]
        for ntable in tables:
            for aent in self.handle_table(ntable):
                yield aent

    def handle_table(self, ntable):
        if False:
            while True:
                i = 10
        ret = []
        if hasattr(ntable, 'hash_mask'):
            hash_size = ntable.hash_mask
            hash_table = ntable.hash_buckets
        elif hasattr(ntable.nht, 'hash_mask'):
            hash_size = ntable.nht.hash_mask
            hash_table = ntable.nht.hash_buckets
        else:
            try:
                hash_size = 1 << ntable.nht.hash_shift
            except OverflowError:
                return []
            hash_table = ntable.nht.hash_buckets
        if not self.addr_space.is_valid_address(hash_table):
            return []
        buckets = obj.Object(theType='Array', offset=hash_table, vm=self.addr_space, targetType='Pointer', count=hash_size)
        if not buckets or hash_size > 50000:
            return []
        for i in range(hash_size):
            if buckets[i]:
                neighbor = obj.Object('neighbour', offset=buckets[i], vm=self.addr_space)
                ret.append(self.walk_neighbor(neighbor))
        return sum(ret, [])

    def walk_neighbor(self, neighbor):
        if False:
            i = 10
            return i + 15
        seen = []
        ret = []
        ctr = 0
        for n in linux_common.walk_internal_list('neighbour', 'next', neighbor):
            if n.obj_offset in seen:
                break
            seen.append(n.obj_offset)
            if ctr > 1024:
                break
            ctr = ctr + 1
            family = n.tbl.family
            if family == socket.AF_INET:
                ip = obj.Object('IpAddress', offset=n.primary_key.obj_offset, vm=self.addr_space).v()
            elif family == socket.AF_INET6:
                ip = obj.Object('Ipv6Address', offset=n.primary_key.obj_offset, vm=self.addr_space).v()
            else:
                ip = '?'
            if n.dev.is_valid():
                mac = ':'.join(['{0:02x}'.format(x) for x in n.ha][:n.dev.addr_len])
                devname = n.dev.name
                ret.append(a_ent(ip, mac, devname))
        return ret

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        for ent in data:
            outfd.write('[{0:42s}] at {1:20s} on {2:s}\n'.format(ent.ip, ent.mac, ent.devname))