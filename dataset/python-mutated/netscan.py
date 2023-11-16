"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import struct, socket
import volatility.debug as debug
import volatility.obj as obj
import volatility.utils as utils
import volatility.plugins.linux.common as linux_common
import volatility.plugins.malware.malfind as malfind
try:
    import yara
    has_yara = True
except ImportError:
    has_yara = False

class linux_netscan(linux_common.AbstractLinuxCommand):
    """Carves for network connection structures"""

    def check_socket_back_pointer(self, i):
        if False:
            return 10
        scomp = self.addr_space.address_compare(i.sk.v(), i.sk.sk_socket.sk.v()) == 0
        zcomp = i.sk.sk_socket.v() == 0
        return scomp or zcomp

    def check_pointers(self, i):
        if False:
            return 10
        ret = self.addr_space.profile.get_symbol_by_address('kernel', i.sk.sk_backlog_rcv.v()) != None
        if ret:
            ret = self.addr_space.profile.get_symbol_by_address('kernel', i.sk.sk_error_report.v()) != None
        return ret

    def check_proto(self, i):
        if False:
            for i in range(10):
                print('nop')
        return i.protocol in ('TCP', 'UDP', 'IP')

    def check_family(self, i):
        if False:
            return 10
        return i.sk.__sk_common.skc_family in (socket.AF_INET, socket.AF_INET6)

    def calculate(self):
        if False:
            return 10
        if not has_yara:
            debug.error('Please install Yara from https://plusvic.github.io/yara/')
        linux_common.set_plugin_members(self)
        if self.addr_space.profile.metadata.get('memory_model', '32bit') == '32bit':
            kernel_start = 3221225472
            pack_size = 4
            pack_fmt = '<I'
        else:
            kernel_start = 18446612132314218496
            pack_size = 8
            pack_fmt = '<Q'
        checks = [self.check_family, self.check_proto, self.check_socket_back_pointer, self.check_pointers]
        destruct_offset = self.addr_space.profile.get_obj_offset('sock', 'sk_destruct')
        func_addr = self.addr_space.profile.get_symbol('inet_sock_destruct')
        vals = struct.pack(pack_fmt, func_addr)
        s = '{ ' + ' '.join(['%.02x' % ord(v) for v in vals]) + ' }'
        rules = yara.compile(sources={'n': 'rule r1 {strings: $a = ' + s + ' condition: $a}'})
        scanner = malfind.DiscontigYaraScanner(rules=rules, address_space=self.addr_space)
        for (_, address) in scanner.scan(start_offset=kernel_start):
            base_address = address - destruct_offset
            i = obj.Object('inet_sock', offset=base_address, vm=self.addr_space)
            valid = True
            for check in checks:
                if check(i) == False:
                    valid = False
                    break
            if valid:
                state = i.state if i.protocol == 'TCP' else ''
                family = i.sk.__sk_common.skc_family
                sport = i.src_port
                dport = i.dst_port
                saddr = i.src_addr
                daddr = i.dst_addr
                if str(saddr) == '0.0.0.0' and str(daddr) == '0.0.0.0' and (sport == 6) and (dport == 0):
                    continue
                yield (i, i.protocol, saddr, sport, daddr, dport, state)

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        for (isock, proto, saddr, sport, daddr, dport, state) in data:
            outfd.write('{6:x} {0:8s} {1:<16}:{2:>5} {3:<16}:{4:>5} {5:<15s}\n'.format(proto, saddr, sport, daddr, dport, state, isock.v()))