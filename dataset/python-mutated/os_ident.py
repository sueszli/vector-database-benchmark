import math
import array
from six.moves import xrange, reduce
from pcapy import lookupdev, open_live
from impacket.ImpactPacket import UDP, TCPOption, Data, TCP, IP, ICMP, Ethernet
from impacket.ImpactDecoder import EthDecoder
from impacket import LOG
g_nmap1_signature_filename = 'nmap-os-fingerprints'
g_nmap2_signature_filename = 'nmap-os-db'

def my_gcd(a, b):
    if False:
        while True:
            i = 10
    if a < b:
        c = a
        a = b
        b = c
    while 0 != b:
        c = a & b
        a = b
        b = c
    return a

class os_id_exception:

    def __init__(self, value):
        if False:
            print('Hello World!')
        self.value = value

    def __str__(self):
        if False:
            return 10
        return repr(self.value)

class os_id_test:

    def __init__(self, id):
        if False:
            while True:
                i = 10
        self.__id = id
        self.__my_packet = None
        self.__result_dict = {}

    def test_id(self):
        if False:
            while True:
                i = 10
        return self.__class__.__name__

    def get_test_packet(self):
        if False:
            i = 10
            return i + 15
        return self.__my_packet.get_packet()

    def set_packet(self, packet):
        if False:
            i = 10
            return i + 15
        self.__my_packet = packet

    def get_packet(self):
        if False:
            print('Hello World!')
        return self.__my_packet

    def process(self, packet):
        if False:
            i = 10
            return i + 15
        pass

    def add_result(self, name, value):
        if False:
            return 10
        self.__result_dict[name] = value

    def get_id(self):
        if False:
            print('Hello World!')
        return self.__id

    def is_mine(self, packet):
        if False:
            return 10
        pass

    def get_result_dict(self):
        if False:
            print('Hello World!')
        return self.__result_dict

    def get_final_result(self):
        if False:
            while True:
                i = 10
        'Returns a string representation of the final result of this test or None if no response was received'
        pass

class icmp_request(os_id_test):
    type_filter = {ICMP.ICMP_ECHO: ICMP.ICMP_ECHOREPLY, ICMP.ICMP_IREQ: ICMP.ICMP_IREQREPLY, ICMP.ICMP_MASKREQ: ICMP.ICMP_MASKREPLY, ICMP.ICMP_TSTAMP: ICMP.ICMP_TSTAMPREPLY}

    def __init__(self, id, addresses, type):
        if False:
            while True:
                i = 10
        os_id_test.__init__(self, id)
        self.e = Ethernet()
        self.i = IP()
        self.icmp = ICMP()
        self.i.set_ip_src(addresses[0])
        self.i.set_ip_dst(addresses[1])
        self.__type = type
        self.icmp.set_icmp_type(type)
        self.e.contains(self.i)
        self.i.contains(self.icmp)
        self.set_packet(self.e)

    def is_mine(self, packet):
        if False:
            for i in range(10):
                print('nop')
        if packet.get_ether_type() != IP.ethertype:
            return 0
        ip = packet.child()
        if not ip or ip.get_ip_p() != ICMP.protocol:
            return 0
        icmp = ip.child()
        if not icmp or icmp.get_icmp_type() != icmp_request.type_filter[self.__type]:
            return 0
        if icmp.get_icmp_id() != self.get_id():
            return 0
        return 1

    def process(self, packet):
        if False:
            while True:
                i = 10
        pass

class nmap2_icmp_echo_probe_1(icmp_request):
    sequence_number = 295
    id = 22136

    def __init__(self, id, addresses):
        if False:
            i = 10
            return i + 15
        icmp_request.__init__(self, id, addresses, ICMP.ICMP_ECHO)
        self.i.set_ip_df(True)
        self.i.set_ip_tos(0)
        self.icmp.set_icmp_code(9)
        self.icmp.set_icmp_seq(nmap2_icmp_echo_probe_1.sequence_number)
        self.i.set_ip_id(nmap2_icmp_echo_probe_1.id)
        self.icmp.set_icmp_id(nmap2_icmp_echo_probe_1.id)
        self.icmp.contains(Data('I' * 120))

    def process(self, packet):
        if False:
            for i in range(10):
                print('nop')
        pass

class nmap2_icmp_echo_probe_2(icmp_request):

    def __init__(self, id, addresses):
        if False:
            while True:
                i = 10
        icmp_request.__init__(self, id, addresses, ICMP.ICMP_ECHO)
        self.i.set_ip_df(False)
        self.i.set_ip_tos(4)
        self.icmp.set_icmp_code(0)
        self.icmp.set_icmp_seq(nmap2_icmp_echo_probe_1.sequence_number + 1)
        self.i.set_ip_id(nmap2_icmp_echo_probe_1.id + 1)
        self.icmp.set_icmp_id(nmap2_icmp_echo_probe_1.id + 1)
        self.icmp.contains(Data('I' * 150))

    def process(self, packet):
        if False:
            return 10
        pass

class udp_closed_probe(os_id_test):
    ip_id = 4660

    def __init__(self, id, addresses, udp_closed):
        if False:
            i = 10
            return i + 15
        os_id_test.__init__(self, id)
        self.e = Ethernet()
        self.i = IP()
        self.u = UDP()
        self.i.set_ip_src(addresses[0])
        self.i.set_ip_dst(addresses[1])
        self.i.set_ip_id(udp_closed_probe.ip_id)
        self.u.set_uh_sport(id)
        self.u.set_uh_dport(udp_closed)
        self.e.contains(self.i)
        self.i.contains(self.u)
        self.set_packet(self.e)

    def is_mine(self, packet):
        if False:
            i = 10
            return i + 15
        if packet.get_ether_type() != IP.ethertype:
            return 0
        ip = packet.child()
        if not ip or ip.get_ip_p() != ICMP.protocol:
            return 0
        icmp = ip.child()
        if not icmp or icmp.get_icmp_type() != ICMP.ICMP_UNREACH:
            return 0
        if icmp.get_icmp_code() != ICMP.ICMP_UNREACH_PORT:
            return 0
        self.err_data = icmp.child()
        if not self.err_data:
            return 0
        return 1

class tcp_probe(os_id_test):

    def __init__(self, id, addresses, tcp_ports, open_port):
        if False:
            return 10
        self.result_string = '[]'
        os_id_test.__init__(self, id)
        self.e = Ethernet()
        self.i = IP()
        self.t = TCP()
        self.i.set_ip_src(addresses[0])
        self.i.set_ip_dst(addresses[1])
        self.i.set_ip_id(8995)
        self.t.set_th_sport(id)
        if open_port:
            self.target_port = tcp_ports[0]
        else:
            self.target_port = tcp_ports[1]
        self.t.set_th_dport(self.target_port)
        self.e.contains(self.i)
        self.i.contains(self.t)
        self.set_packet(self.e)
        self.source_ip = addresses[0]
        self.target_ip = addresses[1]

    def socket_match(self, ip, tcp):
        if False:
            print('Hello World!')
        if ip.get_ip_src() != self.target_ip or tcp.get_th_sport() != self.target_port:
            return 0
        if ip.get_ip_dst() != self.source_ip or tcp.get_th_dport() != self.get_id():
            return 0
        return 1

    def is_mine(self, packet):
        if False:
            print('Hello World!')
        if packet.get_ether_type() != IP.ethertype:
            return 0
        ip = packet.child()
        if not ip or ip.get_ip_p() != TCP.protocol:
            return 0
        tcp = ip.child()
        if self.socket_match(ip, tcp):
            return 1
        return 0

class nmap_tcp_probe(tcp_probe):

    def __init__(self, id, addresses, tcp_ports, open_port, sequence, options):
        if False:
            print('Hello World!')
        tcp_probe.__init__(self, id, addresses, tcp_ports, open_port)
        self.t.set_th_seq(sequence)
        self.set_resp(False)
        for op in options:
            self.t.add_option(op)

    def set_resp(self, resp):
        if False:
            i = 10
            return i + 15
        pass

class nmap1_tcp_probe(nmap_tcp_probe):
    sequence = 33875
    mss = 265
    tcp_options = [TCPOption(TCPOption.TCPOPT_WINDOW, 10), TCPOption(TCPOption.TCPOPT_NOP), TCPOption(TCPOption.TCPOPT_MAXSEG, mss), TCPOption(TCPOption.TCPOPT_TIMESTAMP, 1061109567), TCPOption(TCPOption.TCPOPT_EOL), TCPOption(TCPOption.TCPOPT_EOL)]

    def __init__(self, id, addresses, tcp_ports, open_port):
        if False:
            i = 10
            return i + 15
        nmap_tcp_probe.__init__(self, id, addresses, tcp_ports, open_port, self.sequence, self.tcp_options)

    def set_resp(self, resp):
        if False:
            return 10
        if resp:
            self.add_result('Resp', 'Y')
        else:
            self.add_result('Resp', 'N')

    def process(self, packet):
        if False:
            i = 10
            return i + 15
        ip = packet.child()
        tcp = ip.child()
        self.set_resp(True)
        if ip.get_ip_df():
            self.add_result('DF', 'Y')
        else:
            self.add_result('DF', 'N')
        self.add_result('W', tcp.get_th_win())
        if tcp.get_th_ack() == self.sequence + 1:
            self.add_result('ACK', 'S++')
        elif tcp.get_th_ack() == self.sequence:
            self.add_result('ACK', 'S')
        else:
            self.add_result('ACK', 'O')
        flags = []
        if tcp.get_ECE():
            flags.append('B')
        if tcp.get_URG():
            flags.append('U')
        if tcp.get_ACK():
            flags.append('A')
        if tcp.get_PSH():
            flags.append('P')
        if tcp.get_RST():
            flags.append('R')
        if tcp.get_SYN():
            flags.append('S')
        if tcp.get_FIN():
            flags.append('F')
        self.add_result('FLAGS', flags)
        options = []
        for op in tcp.get_options():
            if op.get_kind() == TCPOption.TCPOPT_EOL:
                options.append('L')
            elif op.get_kind() == TCPOption.TCPOPT_MAXSEG:
                options.append('M')
                if op.get_mss() == self.mss:
                    options.append('E')
            elif op.get_kind() == TCPOption.TCPOPT_NOP:
                options.append('N')
            elif op.get_kind() == TCPOption.TCPOPT_TIMESTAMP:
                options.append('T')
            elif op.get_kind() == TCPOption.TCPOPT_WINDOW:
                options.append('W')
        self.add_result('OPTIONS', options)

    def get_final_result(self):
        if False:
            i = 10
            return i + 15
        return {self.test_id(): self.get_result_dict()}

class nmap2_tcp_probe(nmap_tcp_probe):
    acknowledgment = 404574075

    def __init__(self, id, addresses, tcp_ports, open_port, sequence, options):
        if False:
            return 10
        nmap_tcp_probe.__init__(self, id, addresses, tcp_ports, open_port, sequence, options)
        self.t.set_th_ack(self.acknowledgment)

    def set_resp(self, resp):
        if False:
            for i in range(10):
                print('nop')
        if resp:
            self.add_result('R', 'Y')
        else:
            self.add_result('R', 'N')

    def process(self, packet):
        if False:
            while True:
                i = 10
        ip = packet.child()
        tcp = ip.child()
        self.set_resp(True)
        tests = nmap2_tcp_tests(ip, tcp, self.sequence, self.acknowledgment)
        self.add_result('DF', tests.get_df())
        self.add_result('W', tests.get_win())
        self.add_result('S', tests.get_seq())
        self.add_result('A', tests.get_ack())
        self.add_result('F', tests.get_flags())
        self.add_result('O', tests.get_options())
        self.add_result('Q', tests.get_quirks())

    def get_final_result(self):
        if False:
            for i in range(10):
                print('nop')
        return {self.test_id(): self.get_result_dict()}

class nmap2_ecn_probe(nmap_tcp_probe):
    tcp_options = [TCPOption(TCPOption.TCPOPT_WINDOW, 10), TCPOption(TCPOption.TCPOPT_NOP), TCPOption(TCPOption.TCPOPT_MAXSEG, 1460), TCPOption(TCPOption.TCPOPT_SACK_PERMITTED), TCPOption(TCPOption.TCPOPT_NOP), TCPOption(TCPOption.TCPOPT_NOP)]

    def __init__(self, id, addresses, tcp_ports):
        if False:
            return 10
        nmap_tcp_probe.__init__(self, id, addresses, tcp_ports, 1, 35690, self.tcp_options)
        self.t.set_SYN()
        self.t.set_CWR()
        self.t.set_ECE()
        self.t.set_flags(2048)
        self.t.set_th_urp(63477)
        self.t.set_th_ack(0)
        self.t.set_th_win(3)

    def test_id(self):
        if False:
            print('Hello World!')
        return 'ECN'

    def set_resp(self, resp):
        if False:
            i = 10
            return i + 15
        if resp:
            self.add_result('R', 'Y')
        else:
            self.add_result('R', 'N')

    def process(self, packet):
        if False:
            while True:
                i = 10
        ip = packet.child()
        tcp = ip.child()
        self.set_resp(True)
        tests = nmap2_tcp_tests(ip, tcp, 0, 0)
        self.add_result('DF', tests.get_df())
        self.add_result('W', tests.get_win())
        self.add_result('O', tests.get_options())
        self.add_result('CC', tests.get_cc())
        self.add_result('Q', tests.get_quirks())

    def get_final_result(self):
        if False:
            i = 10
            return i + 15
        return {self.test_id(): self.get_result_dict()}

class nmap2_tcp_tests:

    def __init__(self, ip, tcp, sequence, acknowledgment):
        if False:
            for i in range(10):
                print('nop')
        self.__ip = ip
        self.__tcp = tcp
        self.__sequence = sequence
        self.__acknowledgment = acknowledgment

    def get_df(self):
        if False:
            return 10
        if self.__ip.get_ip_df():
            return 'Y'
        else:
            return 'N'

    def get_win(self):
        if False:
            print('Hello World!')
        return '%X' % self.__tcp.get_th_win()

    def get_ack(self):
        if False:
            return 10
        if self.__tcp.get_th_ack() == self.__sequence + 1:
            return 'S+'
        elif self.__tcp.get_th_ack() == self.__sequence:
            return 'S'
        elif self.__tcp.get_th_ack() == 0:
            return 'Z'
        else:
            return 'O'

    def get_seq(self):
        if False:
            print('Hello World!')
        if self.__tcp.get_th_seq() == self.__acknowledgment + 1:
            return 'A+'
        elif self.__tcp.get_th_seq() == self.__acknowledgment:
            return 'A'
        elif self.__tcp.get_th_seq() == 0:
            return 'Z'
        else:
            return 'O'

    def get_flags(self):
        if False:
            return 10
        flags = ''
        if self.__tcp.get_ECE():
            flags += 'E'
        if self.__tcp.get_URG():
            flags += 'U'
        if self.__tcp.get_ACK():
            flags += 'A'
        if self.__tcp.get_PSH():
            flags += 'P'
        if self.__tcp.get_RST():
            flags += 'R'
        if self.__tcp.get_SYN():
            flags += 'S'
        if self.__tcp.get_FIN():
            flags += 'F'
        return flags

    def get_options(self):
        if False:
            for i in range(10):
                print('nop')
        options = ''
        for op in self.__tcp.get_options():
            if op.get_kind() == TCPOption.TCPOPT_EOL:
                options += 'L'
            elif op.get_kind() == TCPOption.TCPOPT_MAXSEG:
                options += 'M%X' % op.get_mss()
            elif op.get_kind() == TCPOption.TCPOPT_NOP:
                options += 'N'
            elif op.get_kind() == TCPOption.TCPOPT_TIMESTAMP:
                options += 'T%i%i' % (int(op.get_ts() != 0), int(op.get_ts_echo() != 0))
            elif op.get_kind() == TCPOption.TCPOPT_WINDOW:
                options += 'W%X' % op.get_shift_cnt()
            elif op.get_kind() == TCPOption.TCPOPT_SACK_PERMITTED:
                options += 'S'
        return options

    def get_cc(self):
        if False:
            print('Hello World!')
        (ece, cwr) = (self.__tcp.get_ECE(), self.__tcp.get_CWR())
        if ece and (not cwr):
            return 'Y'
        elif not ece and (not cwr):
            return 'N'
        elif ece and cwr:
            return 'S'
        else:
            return 'O'

    def get_quirks(self):
        if False:
            return 10
        quirks = ''
        if self.__tcp.get_th_flags() >> 8 & 15 != 0:
            quirks += 'R'
        if self.__tcp.get_URG() == 0 and self.__tcp.get_th_urp() != 0:
            quirks += 'U'
        return quirks

class nmap2_tcp_probe_2_6(nmap2_tcp_probe):
    sequence = 33875
    mss = 265
    tcp_options = [TCPOption(TCPOption.TCPOPT_WINDOW, 10), TCPOption(TCPOption.TCPOPT_NOP), TCPOption(TCPOption.TCPOPT_MAXSEG, mss), TCPOption(TCPOption.TCPOPT_TIMESTAMP, 4294967295), TCPOption(TCPOption.TCPOPT_SACK_PERMITTED)]

    def __init__(self, id, addresses, tcp_ports, open_port):
        if False:
            for i in range(10):
                print('nop')
        nmap2_tcp_probe.__init__(self, id, addresses, tcp_ports, open_port, self.sequence, self.tcp_options)

class nmap2_tcp_probe_7(nmap2_tcp_probe):
    sequence = 33875
    mss = 265
    tcp_options = [TCPOption(TCPOption.TCPOPT_WINDOW, 15), TCPOption(TCPOption.TCPOPT_NOP), TCPOption(TCPOption.TCPOPT_MAXSEG, mss), TCPOption(TCPOption.TCPOPT_TIMESTAMP, 4294967295), TCPOption(TCPOption.TCPOPT_SACK_PERMITTED)]

    def __init__(self, id, addresses, tcp_ports, open_port):
        if False:
            print('Hello World!')
        nmap2_tcp_probe.__init__(self, id, addresses, tcp_ports, open_port, self.sequence, self.tcp_options)

class nmap_port_unreachable(udp_closed_probe):

    def __init__(self, id, addresses, ports):
        if False:
            print('Hello World!')
        udp_closed_probe.__init__(self, id, addresses, ports[2])
        self.set_resp(False)

    def test_id(self):
        if False:
            while True:
                i = 10
        pass

    def set_resp(self, resp):
        if False:
            print('Hello World!')
        pass

    def process(self, packet):
        if False:
            i = 10
            return i + 15
        pass

class nmap1_port_unreachable(nmap_port_unreachable):

    def __init__(self, id, addresses, ports):
        if False:
            while True:
                i = 10
        nmap_port_unreachable.__init__(self, id, addresses, ports)
        self.u.contains(Data('A' * 300))

    def test_id(self):
        if False:
            return 10
        return 'PU'

    def set_resp(self, resp):
        if False:
            return 10
        if resp:
            self.add_result('Resp', 'Y')
        else:
            self.add_result('Resp', 'N')

    def process(self, packet):
        if False:
            for i in range(10):
                print('nop')
        ip_orig = self.err_data
        if ip_orig.get_ip_p() != UDP.protocol:
            return
        udp = ip_orig.child()
        if not udp:
            return
        ip = packet.child()
        self.set_resp(True)
        if ip.get_ip_df():
            self.add_result('DF', 'Y')
        else:
            self.add_result('DF', 'N')
        self.add_result('TOS', ip.get_ip_tos())
        self.add_result('IPLEN', ip.get_ip_len())
        self.add_result('RIPTL', ip_orig.get_ip_len())
        recv_ip_id = ip_orig.get_ip_id()
        if 0 == recv_ip_id:
            self.add_result('RID', '0')
        elif udp_closed_probe.ip_id == recv_ip_id:
            self.add_result('RID', 'E')
        else:
            self.add_result('RID', 'F')
        ip_sum = ip_orig.get_ip_sum()
        ip_orig.set_ip_sum(0)
        checksum = ip_orig.compute_checksum(ip_orig.get_bytes())
        if 0 == checksum:
            self.add_result('RIPCK', '0')
        elif checksum == ip_sum:
            self.add_result('RIPCK', 'E')
        else:
            self.add_result('RIPCK', 'F')
        udp_sum = udp.get_uh_sum()
        udp.set_uh_sum(0)
        udp.auto_checksum = 1
        udp.calculate_checksum()
        if 0 == udp_sum:
            self.add_result('UCK', '0')
        elif self.u.get_uh_sum() == udp_sum:
            self.add_result('UCK', 'E')
        else:
            self.add_result('UCK', 'F')
        self.add_result('ULEN', udp.get_uh_ulen())
        if ip.child().child().child().child() == udp.child():
            self.add_result('DAT', 'E')
        else:
            self.add_result('DAT', 'F')

    def get_final_result(self):
        if False:
            print('Hello World!')
        return {self.test_id(): self.get_result_dict()}

class nmap2_port_unreachable(nmap_port_unreachable):

    def __init__(self, id, addresses, ports):
        if False:
            print('Hello World!')
        nmap_port_unreachable.__init__(self, id, addresses, ports)
        self.u.contains(Data('C' * 300))
        self.i.set_ip_id(4162)

    def test_id(self):
        if False:
            for i in range(10):
                print('nop')
        return 'U1'

    def set_resp(self, resp):
        if False:
            print('Hello World!')
        if resp:
            self.add_result('R', 'Y')
        else:
            self.add_result('R', 'N')

    def process(self, packet):
        if False:
            print('Hello World!')
        ip_orig = self.err_data
        if ip_orig.get_ip_p() != UDP.protocol:
            return
        udp = ip_orig.child()
        if not udp:
            return
        ip = packet.child()
        icmp = ip.child()
        if ip.get_ip_df():
            self.add_result('DF', 'Y')
        else:
            self.add_result('DF', 'N')
        self.add_result('TOS', '%X' % ip.get_ip_tos())
        self.add_result('IPL', '%X' % ip.get_ip_len())
        self.add_result('UN', '%X' % icmp.get_icmp_void())
        if ip_orig.get_ip_len() == 328:
            self.add_result('RIPL', 'G')
        else:
            self.add_result('RIPL', '%X' % ip_orig.get_ip_len())
        if 4162 == ip_orig.get_ip_id():
            self.add_result('RID', 'G')
        else:
            self.add_result('RID', '%X' % ip_orig.get_ip_id())
        ip_sum = ip_orig.get_ip_sum()
        ip_orig.set_ip_sum(0)
        checksum = ip_orig.compute_checksum(ip_orig.get_bytes())
        if 0 == checksum:
            self.add_result('RIPCK', 'Z')
        elif checksum == ip_sum:
            self.add_result('RIPCK', 'G')
        else:
            self.add_result('RIPCK', 'I')
        udp_sum = udp.get_uh_sum()
        udp.set_uh_sum(0)
        udp.auto_checksum = 1
        udp.calculate_checksum()
        if self.u.get_uh_sum() == udp_sum:
            self.add_result('RUCK', 'G')
        else:
            self.add_result('RUCK', '%X' % udp_sum)
        if udp.get_uh_ulen() == 308:
            self.add_result('RUL', 'G')
        else:
            self.add_result('RUL', '%X' % udp.get_uh_ulen())
        if ip.child().child().child().child() == udp.child():
            self.add_result('RUD', 'G')
        else:
            self.add_result('RUD', 'I')

    def get_final_result(self):
        if False:
            print('Hello World!')
        return {self.test_id(): self.get_result_dict()}

class OS_ID:

    def __init__(self, target, ports):
        if False:
            print('Hello World!')
        pcap_dev = lookupdev()
        self.p = open_live(pcap_dev, 600, 0, 3000)
        self.__source = self.p.getlocalip()
        self.__target = target
        self.p.setfilter('src host %s and dst host %s' % (target, self.__source), 1, 4294967040)
        self.p.setmintocopy(10)
        self.decoder = EthDecoder()
        self.tests_sent = []
        self.outstanding_count = 0
        self.results = {}
        self.current_id = 12345
        self.__ports = ports

    def releasePcap(self):
        if False:
            print('Hello World!')
        if not self.p is None:
            self.p.close()

    def get_new_id(self):
        if False:
            for i in range(10):
                print('nop')
        id = self.current_id
        self.current_id += 1
        self.current_id &= 65535
        return id

    def send_tests(self, tests):
        if False:
            return 10
        self.outstanding_count = 0
        for t_class in tests:
            if t_class.__init__.im_func.func_code.co_argcount == 4:
                test = t_class(self.get_new_id(), [self.__source, self.__target], self.__ports)
            else:
                test = t_class(self.get_new_id(), [self.__source, self.__target])
            self.p.sendpacket(test.get_test_packet())
            self.outstanding_count += 1
            self.tests_sent.append(test)
            while self.p.readready():
                self.p.dispatch(1, self.packet_handler)
        while self.outstanding_count > 0:
            data = self.p.next()[0]
            if data:
                self.packet_handler(0, data)
            else:
                break

    def run(self):
        if False:
            while True:
                i = 10
        pass

    def get_source(self):
        if False:
            while True:
                i = 10
        return self.__source

    def get_target(self):
        if False:
            while True:
                i = 10
        return self.__target

    def get_ports(self):
        if False:
            return 10
        return self.__ports

    def packet_handler(self, len, data):
        if False:
            while True:
                i = 10
        packet = self.decoder.decode(data)
        for t in self.tests_sent:
            if t.is_mine(packet):
                t.process(packet)
                self.outstanding_count -= 1

class nmap1_tcp_open_1(nmap1_tcp_probe):

    def __init__(self, id, addresses, tcp_ports):
        if False:
            for i in range(10):
                print('nop')
        nmap1_tcp_probe.__init__(self, id, addresses, tcp_ports, 1)
        self.t.set_ECE()
        self.t.set_SYN()

    def test_id(self):
        if False:
            while True:
                i = 10
        return 'T1'

    def is_mine(self, packet):
        if False:
            return 10
        if tcp_probe.is_mine(self, packet):
            ip = packet.child()
            if not ip:
                return 0
            tcp = ip.child()
            if not tcp:
                return 0
            if tcp.get_SYN() and tcp.get_ACK():
                return 1
            else:
                return 0
        else:
            return 0

class nmap1_tcp_open_2(nmap1_tcp_probe):

    def __init__(self, id, addresses, tcp_ports):
        if False:
            for i in range(10):
                print('nop')
        nmap1_tcp_probe.__init__(self, id, addresses, tcp_ports, 1)

    def test_id(self):
        if False:
            print('Hello World!')
        return 'T2'

class nmap2_tcp_open_2(nmap2_tcp_probe_2_6):

    def __init__(self, id, addresses, tcp_ports):
        if False:
            print('Hello World!')
        nmap2_tcp_probe_2_6.__init__(self, id, addresses, tcp_ports, 1)
        self.i.set_ip_df(1)
        self.t.set_th_win(128)

    def test_id(self):
        if False:
            while True:
                i = 10
        return 'T2'

class nmap1_tcp_open_3(nmap1_tcp_probe):

    def __init__(self, id, addresses, tcp_ports):
        if False:
            i = 10
            return i + 15
        nmap1_tcp_probe.__init__(self, id, addresses, tcp_ports, 1)
        self.t.set_SYN()
        self.t.set_FIN()
        self.t.set_URG()
        self.t.set_PSH()

    def test_id(self):
        if False:
            while True:
                i = 10
        return 'T3'

class nmap2_tcp_open_3(nmap2_tcp_probe_2_6):

    def __init__(self, id, addresses, tcp_ports):
        if False:
            return 10
        nmap2_tcp_probe_2_6.__init__(self, id, addresses, tcp_ports, 1)
        self.t.set_SYN()
        self.t.set_FIN()
        self.t.set_URG()
        self.t.set_PSH()
        self.t.set_th_win(256)
        self.i.set_ip_df(0)

    def test_id(self):
        if False:
            while True:
                i = 10
        return 'T3'

class nmap1_tcp_open_4(nmap1_tcp_probe):

    def __init__(self, id, addresses, tcp_ports):
        if False:
            return 10
        nmap1_tcp_probe.__init__(self, id, addresses, tcp_ports, 1)
        self.t.set_ACK()

    def test_id(self):
        if False:
            return 10
        return 'T4'

class nmap2_tcp_open_4(nmap2_tcp_probe_2_6):

    def __init__(self, id, addresses, tcp_ports):
        if False:
            print('Hello World!')
        nmap2_tcp_probe_2_6.__init__(self, id, addresses, tcp_ports, 1)
        self.t.set_ACK()
        self.i.set_ip_df(1)
        self.t.set_th_win(1024)

    def test_id(self):
        if False:
            i = 10
            return i + 15
        return 'T4'

class nmap1_seq(nmap1_tcp_probe):
    SEQ_UNKNOWN = 0
    SEQ_64K = 1
    SEQ_TD = 2
    SEQ_RI = 4
    SEQ_TR = 8
    SEQ_i800 = 16
    SEQ_CONSTANT = 32
    TS_SEQ_UNKNOWN = 0
    TS_SEQ_ZERO = 1
    TS_SEQ_2HZ = 2
    TS_SEQ_100HZ = 3
    TS_SEQ_1000HZ = 4
    TS_SEQ_UNSUPPORTED = 5
    IPID_SEQ_UNKNOWN = 0
    IPID_SEQ_INCR = 1
    IPID_SEQ_BROKEN_INCR = 2
    IPID_SEQ_RPI = 3
    IPID_SEQ_RD = 4
    IPID_SEQ_CONSTANT = 5
    IPID_SEQ_ZERO = 6

    def __init__(self, id, addresses, tcp_ports):
        if False:
            print('Hello World!')
        nmap1_tcp_probe.__init__(self, id, addresses, tcp_ports, 1)
        self.t.set_SYN()
        self.t.set_th_seq(id)

    def process(self, p):
        if False:
            for i in range(10):
                print('nop')
        raise Exception('Method process is meaningless for class %s.' % self.__class__.__name__)

class nmap2_seq(nmap2_tcp_probe):
    TS_SEQ_UNKNOWN = 0
    TS_SEQ_ZERO = 1
    TS_SEQ_UNSUPPORTED = 5
    IPID_SEQ_UNKNOWN = 0
    IPID_SEQ_INCR = 1
    IPID_SEQ_BROKEN_INCR = 2
    IPID_SEQ_RPI = 3
    IPID_SEQ_RD = 4
    IPID_SEQ_CONSTANT = 5
    IPID_SEQ_ZERO = 6

    def __init__(self, id, addresses, tcp_ports, options):
        if False:
            return 10
        nmap2_tcp_probe.__init__(self, id, addresses, tcp_ports, 1, id, options)
        self.t.set_SYN()

    def process(self, p):
        if False:
            while True:
                i = 10
        raise Exception('Method process is meaningless for class %s.' % self.__class__.__name__)

class nmap2_seq_1(nmap2_seq):
    tcp_options = [TCPOption(TCPOption.TCPOPT_WINDOW, 10), TCPOption(TCPOption.TCPOPT_NOP), TCPOption(TCPOption.TCPOPT_MAXSEG, 1460), TCPOption(TCPOption.TCPOPT_TIMESTAMP, 4294967295), TCPOption(TCPOption.TCPOPT_SACK_PERMITTED)]

    def __init__(self, id, addresses, tcp_ports):
        if False:
            return 10
        nmap2_seq.__init__(self, id, addresses, tcp_ports, self.tcp_options)
        self.t.set_th_win(1)

class nmap2_seq_2(nmap2_seq):
    tcp_options = [TCPOption(TCPOption.TCPOPT_MAXSEG, 1400), TCPOption(TCPOption.TCPOPT_WINDOW, 0), TCPOption(TCPOption.TCPOPT_SACK_PERMITTED), TCPOption(TCPOption.TCPOPT_TIMESTAMP, 4294967295), TCPOption(TCPOption.TCPOPT_EOL)]

    def __init__(self, id, addresses, tcp_ports):
        if False:
            i = 10
            return i + 15
        nmap2_seq.__init__(self, id, addresses, tcp_ports, self.tcp_options)
        self.t.set_th_win(63)

class nmap2_seq_3(nmap2_seq):
    tcp_options = [TCPOption(TCPOption.TCPOPT_TIMESTAMP, 4294967295), TCPOption(TCPOption.TCPOPT_NOP), TCPOption(TCPOption.TCPOPT_NOP), TCPOption(TCPOption.TCPOPT_WINDOW, 5), TCPOption(TCPOption.TCPOPT_NOP), TCPOption(TCPOption.TCPOPT_MAXSEG, 640)]

    def __init__(self, id, addresses, tcp_ports):
        if False:
            while True:
                i = 10
        nmap2_seq.__init__(self, id, addresses, tcp_ports, self.tcp_options)
        self.t.set_th_win(4)

class nmap2_seq_4(nmap2_seq):
    tcp_options = [TCPOption(TCPOption.TCPOPT_SACK_PERMITTED), TCPOption(TCPOption.TCPOPT_TIMESTAMP, 4294967295), TCPOption(TCPOption.TCPOPT_WINDOW, 10), TCPOption(TCPOption.TCPOPT_EOL)]

    def __init__(self, id, addresses, tcp_ports):
        if False:
            while True:
                i = 10
        nmap2_seq.__init__(self, id, addresses, tcp_ports, self.tcp_options)
        self.t.set_th_win(4)

class nmap2_seq_5(nmap2_seq):
    tcp_options = [TCPOption(TCPOption.TCPOPT_MAXSEG, 536), TCPOption(TCPOption.TCPOPT_SACK_PERMITTED), TCPOption(TCPOption.TCPOPT_TIMESTAMP, 4294967295), TCPOption(TCPOption.TCPOPT_WINDOW, 10), TCPOption(TCPOption.TCPOPT_EOL)]

    def __init__(self, id, addresses, tcp_ports):
        if False:
            return 10
        nmap2_seq.__init__(self, id, addresses, tcp_ports, self.tcp_options)
        self.t.set_th_win(16)

class nmap2_seq_6(nmap2_seq):
    tcp_options = [TCPOption(TCPOption.TCPOPT_MAXSEG, 265), TCPOption(TCPOption.TCPOPT_SACK_PERMITTED), TCPOption(TCPOption.TCPOPT_TIMESTAMP, 4294967295)]

    def __init__(self, id, addresses, tcp_ports):
        if False:
            for i in range(10):
                print('nop')
        nmap2_seq.__init__(self, id, addresses, tcp_ports, self.tcp_options)
        self.t.set_th_win(512)

class nmap1_seq_container(os_id_test):

    def __init__(self, num_seq_samples, responses, seq_diffs, ts_diffs, time_diffs):
        if False:
            print('Hello World!')
        os_id_test.__init__(self, 0)
        self.num_seq_samples = num_seq_samples
        self.seq_responses = responses
        self.seq_num_responses = len(responses)
        self.seq_diffs = seq_diffs
        self.ts_diffs = ts_diffs
        self.time_diffs = time_diffs
        self.pre_ts_seqclass = nmap1_seq.TS_SEQ_UNKNOWN

    def test_id(self):
        if False:
            print('Hello World!')
        return 'TSEQ'

    def set_ts_seqclass(self, ts_seqclass):
        if False:
            print('Hello World!')
        self.pre_ts_seqclass = ts_seqclass

    def process(self):
        if False:
            i = 10
            return i + 15
        ipid_seqclass = self.ipid_sequence()
        if nmap1_seq.TS_SEQ_UNKNOWN != self.pre_ts_seqclass:
            ts_seqclass = self.pre_ts_seqclass
        else:
            ts_seqclass = self.ts_sequence()
        if self.seq_num_responses >= 4:
            seq_seqclass = self.seq_sequence()
            if nmap1_seq.SEQ_UNKNOWN != seq_seqclass:
                self.add_seqclass(seq_seqclass)
            if nmap1_seq.IPID_SEQ_UNKNOWN != ipid_seqclass:
                self.add_ipidclass(ipid_seqclass)
            if nmap1_seq.TS_SEQ_UNKNOWN != ts_seqclass:
                self.add_tsclass(ts_seqclass)
        else:
            LOG.error('Insufficient responses for TCP sequencing (%d out of %d), OS detection may be less accurate.' % (self.seq_num_responses, self.num_seq_samples))

    def get_final_result(self):
        if False:
            print('Hello World!')
        'Returns a string representation of the final result of this test or None if no response was received'
        return {self.test_id(): self.get_result_dict()}

    def ipid_sequence(self):
        if False:
            for i in range(10):
                print('nop')
        if self.seq_num_responses < 2:
            return nmap1_seq.IPID_SEQ_UNKNOWN
        ipid_diffs = array.array('H', [0] * (self.seq_num_responses - 1))
        null_ipids = 1
        for i in xrange(1, self.seq_num_responses):
            prev_ipid = self.seq_responses[i - 1].get_ipid()
            cur_ipid = self.seq_responses[i].get_ipid()
            if cur_ipid < prev_ipid and (cur_ipid > 500 or prev_ipid < 65000):
                return nmap1_seq.IPID_SEQ_RD
            if prev_ipid != 0 or cur_ipid != 0:
                null_ipids = 0
            ipid_diffs[i - 1] = abs(cur_ipid - prev_ipid)
        if null_ipids:
            return nmap1_seq.IPID_SEQ_ZERO
        for i in xrange(0, self.seq_num_responses - 1):
            if ipid_diffs[i] > 1000:
                return nmap1_seq.IPID_SEQ_RPI
            if ipid_diffs[i] == 0:
                return nmap1_seq.IPID_SEQ_CONSTANT
        is_incremental = 1
        is_ms = 1
        for i in xrange(0, self.seq_num_responses - 1):
            if ipid_diffs[i] == 1:
                return nmap1_seq.IPID_SEQ_INCR
            if is_ms and ipid_diffs[i] < 2560 and (ipid_diffs[i] % 256 != 0):
                is_ms = 0
            if ipid_diffs[i] > 9:
                is_incremental = 0
        if is_ms:
            return nmap1_seq.IPID_SEQ_BROKEN_INCR
        if is_incremental:
            return nmap1_seq.IPID_SEQ_INCR
        return nmap1_seq.IPID_SEQ_UNKNOWN

    def ts_sequence(self):
        if False:
            i = 10
            return i + 15
        if self.seq_num_responses < 2:
            return nmap1_seq.TS_SEQ_UNKNOWN
        avg_freq = 0.0
        for i in xrange(0, self.seq_num_responses - 1):
            dhz = self.ts_diffs[i] / self.time_diffs[i]
            avg_freq += dhz / (self.seq_num_responses - 1)
        LOG.info('The avg TCP TS HZ is: %f' % avg_freq)
        if 0 < avg_freq < 3.9:
            return nmap1_seq.TS_SEQ_2HZ
        if 85 < avg_freq < 115:
            return nmap1_seq.TS_SEQ_100HZ
        if 900 < avg_freq < 1100:
            return nmap1_seq.TS_SEQ_1000HZ
        return nmap1_seq.TS_SEQ_UNKNOWN

    def seq_sequence(self):
        if False:
            while True:
                i = 10
        self.seq_gcd = reduce(my_gcd, self.seq_diffs)
        avg_incr = 0
        seqclass = nmap1_seq.SEQ_UNKNOWN
        if 0 != self.seq_gcd:
            map(lambda x, gcd=self.seq_gcd: x / gcd, self.seq_diffs)
            for i in xrange(0, self.seq_num_responses - 1):
                if abs(self.seq_responses[i + 1].get_seq() - self.seq_responses[i].get_seq()) > 50000000:
                    seqclass = nmap1_seq.SEQ_TR
                    self.index = 9999999
                    break
                avg_incr += self.seq_diffs[i]
        if 0 == self.seq_gcd:
            seqclass = nmap1_seq.SEQ_CONSTANT
            self.index = 0
        elif 0 == self.seq_gcd % 64000:
            seqclass = nmap1_seq.SEQ_64K
            self.index = 1
        elif 0 == self.seq_gcd % 800:
            seqclass = nmap1_seq.SEQ_i800
            self.index = 10
        elif nmap1_seq.SEQ_UNKNOWN == seqclass:
            avg_incr = int(0.5 + avg_incr / (self.seq_num_responses - 1))
            sum_incr = 0.0
            for i in range(0, self.seq_num_responses - 1):
                d = abs(self.seq_diffs[i] - avg_incr)
                sum_incr += float(d * d)
            sum_incr /= self.seq_num_responses - 1
            self.index = int(0.5 + math.sqrt(sum_incr))
            if self.index < 75:
                seqclass = nmap1_seq.SEQ_TD
            else:
                seqclass = nmap1_seq.SEQ_RI
        return seqclass
    seqclasses = {nmap1_seq.SEQ_64K: '64K', nmap1_seq.SEQ_TD: 'TD', nmap1_seq.SEQ_RI: 'RI', nmap1_seq.SEQ_TR: 'TR', nmap1_seq.SEQ_i800: 'i800', nmap1_seq.SEQ_CONSTANT: 'C'}

    def add_seqclass(self, id):
        if False:
            for i in range(10):
                print('nop')
        self.add_result('CLASS', nmap1_seq_container.seqclasses[id])
        if nmap1_seq.SEQ_CONSTANT == id:
            self.add_result('VAL', '%i' % self.seq_responses[0].get_seq())
        elif id in (nmap1_seq.SEQ_TD, nmap1_seq.SEQ_RI):
            self.add_result('GCD', '%i' % self.seq_gcd)
            self.add_result('SI', '%i' % self.index)
    tsclasses = {nmap1_seq.TS_SEQ_ZERO: '0', nmap1_seq.TS_SEQ_2HZ: '2HZ', nmap1_seq.TS_SEQ_100HZ: '100HZ', nmap1_seq.TS_SEQ_1000HZ: '1000HZ', nmap1_seq.TS_SEQ_UNSUPPORTED: 'U'}

    def add_tsclass(self, id):
        if False:
            for i in range(10):
                print('nop')
        self.add_result('TS', nmap1_seq_container.tsclasses[id])
    ipidclasses = {nmap1_seq.IPID_SEQ_INCR: 'I', nmap1_seq.IPID_SEQ_BROKEN_INCR: 'BI', nmap1_seq.IPID_SEQ_RPI: 'RPI', nmap1_seq.IPID_SEQ_RD: 'RD', nmap1_seq.IPID_SEQ_CONSTANT: 'C', nmap1_seq.IPID_SEQ_ZERO: 'Z'}

    def add_ipidclass(self, id):
        if False:
            i = 10
            return i + 15
        self.add_result('IPID', nmap1_seq_container.ipidclasses[id])

class nmap2_seq_container(os_id_test):

    def __init__(self, num_seq_samples, responses, seq_diffs, ts_diffs, time_diffs):
        if False:
            return 10
        os_id_test.__init__(self, 0)
        self.num_seq_samples = num_seq_samples
        self.seq_responses = responses
        self.seq_num_responses = len(responses)
        self.seq_diffs = seq_diffs
        self.ts_diffs = ts_diffs
        self.time_diffs = time_diffs
        self.pre_ts_seqclass = nmap2_seq.TS_SEQ_UNKNOWN

    def test_id(self):
        if False:
            i = 10
            return i + 15
        return 'SEQ'

    def set_ts_seqclass(self, ts_seqclass):
        if False:
            for i in range(10):
                print('nop')
        self.pre_ts_seqclass = ts_seqclass

    def process(self):
        if False:
            while True:
                i = 10
        if self.seq_num_responses >= 4:
            self.calc_ti()
            self.calc_ts()
            self.calc_sp()
        else:
            self.add_result('R', 'N')
            LOG.error('Insufficient responses for TCP sequencing (%d out of %d), OS detection may be less accurate.' % (self.seq_num_responses, self.num_seq_samples))

    def get_final_result(self):
        if False:
            for i in range(10):
                print('nop')
        return {self.test_id(): self.get_result_dict()}

    def calc_ti(self):
        if False:
            for i in range(10):
                print('nop')
        if self.seq_num_responses < 2:
            return
        ipidclasses = {nmap2_seq.IPID_SEQ_INCR: 'I', nmap2_seq.IPID_SEQ_BROKEN_INCR: 'BI', nmap2_seq.IPID_SEQ_RPI: 'RI', nmap2_seq.IPID_SEQ_RD: 'RD', nmap2_seq.IPID_SEQ_CONSTANT: 'C', nmap2_seq.IPID_SEQ_ZERO: 'Z'}
        ipid_diffs = array.array('H', [0] * (self.seq_num_responses - 1))
        null_ipids = 1
        for i in xrange(1, self.seq_num_responses):
            prev_ipid = self.seq_responses[i - 1].get_ipid()
            cur_ipid = self.seq_responses[i].get_ipid()
            if prev_ipid != 0 or cur_ipid != 0:
                null_ipids = 0
            if prev_ipid <= cur_ipid:
                ipid_diffs[i - 1] = cur_ipid - prev_ipid
            else:
                ipid_diffs[i - 1] = cur_ipid - prev_ipid + 65536 & 65535
            if self.seq_num_responses > 2 and ipid_diffs[i - 1] > 20000:
                self.add_result('TI', ipidclasses[nmap2_seq.IPID_SEQ_RD])
                return
        if null_ipids:
            self.add_result('TI', ipidclasses[nmap2_seq.IPID_SEQ_ZERO])
            return
        all_zero = 1
        for i in xrange(0, self.seq_num_responses - 1):
            if ipid_diffs[i] != 0:
                all_zero = 0
                break
        if all_zero:
            self.add_result('TI', ipidclasses[nmap2_seq.IPID_SEQ_CONSTANT])
            return
        for i in xrange(0, self.seq_num_responses - 1):
            if ipid_diffs[i] > 1000 and (ipid_diffs[i] % 256 != 0 or (ipid_diffs[i] % 256 == 0 and ipid_diffs[i] >= 25600)):
                self.add_result('TI', ipidclasses[nmap2_seq.IPID_SEQ_RPI])
                return
        is_incremental = 1
        is_ms = 1
        for i in xrange(0, self.seq_num_responses - 1):
            if is_ms and (ipid_diffs[i] > 5120 or ipid_diffs[i] % 256 != 0):
                is_ms = 0
            if is_incremental and ipid_diffs[i] > 9:
                is_incremental = 0
        if is_ms:
            self.add_result('TI', ipidclasses[nmap2_seq.IPID_SEQ_BROKEN_INCR])
        elif is_incremental:
            self.add_result('TI', ipidclasses[nmap2_seq.IPID_SEQ_INCR])

    def calc_ts(self):
        if False:
            for i in range(10):
                print('nop')
        if self.pre_ts_seqclass == nmap2_seq.TS_SEQ_ZERO:
            self.add_result('TS', '0')
        elif self.pre_ts_seqclass == nmap2_seq.TS_SEQ_UNSUPPORTED:
            self.add_result('TS', 'U')
        elif self.seq_num_responses < 2:
            return
        avg_freq = 0.0
        for i in xrange(0, self.seq_num_responses - 1):
            dhz = self.ts_diffs[i] / self.time_diffs[i]
            avg_freq += dhz / (self.seq_num_responses - 1)
        LOG.info('The avg TCP TS HZ is: %f' % avg_freq)
        if avg_freq <= 5.66:
            self.add_result('TS', '1')
        elif 70 < avg_freq <= 150:
            self.add_result('TS', '7')
        elif 150 < avg_freq <= 350:
            self.add_result('TS', '8')
        else:
            ts = int(round(0.5 + math.log(avg_freq) / math.log(2)))
            self.add_result('TS', '%X' % ts)

    def calc_sp(self):
        if False:
            return 10
        seq_gcd = reduce(my_gcd, self.seq_diffs)
        seq_avg_rate = 0.0
        for i in xrange(0, self.seq_num_responses - 1):
            seq_avg_rate += self.seq_diffs[i] / self.time_diffs[i]
        seq_avg_rate /= self.seq_num_responses - 1
        seq_rate = seq_avg_rate
        si_index = 0
        seq_stddev = 0
        if 0 == seq_gcd:
            seq_rate = 0
        else:
            seq_rate = int(round(0.5 + math.log(seq_rate) / math.log(2) * 8))
            div_gcd = 1
            if seq_gcd > 9:
                div_gcd = seq_gcd
            for i in xrange(0, self.seq_num_responses - 1):
                rtmp = self.seq_diffs[i] / self.time_diffs[i] / div_gcd - seq_avg_rate / div_gcd
                seq_stddev += rtmp * rtmp
            seq_stddev /= self.seq_num_responses - 2
            seq_stddev = math.sqrt(seq_stddev)
            if seq_stddev <= 1:
                si_index = 0
            else:
                si_index = int(round(0.5 + math.log(seq_stddev) / math.log(2) * 8.0))
        self.add_result('SP', '%X' % si_index)
        self.add_result('GCD', '%X' % seq_gcd)
        self.add_result('ISR', '%X' % seq_rate)

class nmap2_ops_container(os_id_test):

    def __init__(self, responses):
        if False:
            for i in range(10):
                print('nop')
        os_id_test.__init__(self, 0)
        self.seq_responses = responses
        self.seq_num_responses = len(responses)

    def test_id(self):
        if False:
            while True:
                i = 10
        return 'OPS'

    def process(self):
        if False:
            return 10
        if self.seq_num_responses != 6:
            self.add_result('R', 'N')
            return
        for i in xrange(0, self.seq_num_responses):
            tests = nmap2_tcp_tests(self.seq_responses[i].get_ip(), self.seq_responses[i].get_tcp(), 0, 0)
            self.add_result('O%i' % (i + 1), tests.get_options())

    def get_final_result(self):
        if False:
            while True:
                i = 10
        if not self.get_result_dict():
            return None
        else:
            return {self.test_id(): self.get_result_dict()}

class nmap2_win_container(os_id_test):

    def __init__(self, responses):
        if False:
            for i in range(10):
                print('nop')
        os_id_test.__init__(self, 0)
        self.seq_responses = responses
        self.seq_num_responses = len(responses)

    def test_id(self):
        if False:
            print('Hello World!')
        return 'WIN'

    def process(self):
        if False:
            i = 10
            return i + 15
        if self.seq_num_responses != 6:
            self.add_result('R', 'N')
            return
        for i in xrange(0, self.seq_num_responses):
            tests = nmap2_tcp_tests(self.seq_responses[i].get_ip(), self.seq_responses[i].get_tcp(), 0, 0)
            self.add_result('W%i' % (i + 1), tests.get_win())

    def get_final_result(self):
        if False:
            return 10
        if not self.get_result_dict():
            return None
        else:
            return {self.test_id(): self.get_result_dict()}

class nmap2_t1_container(os_id_test):

    def __init__(self, responses, seq_base):
        if False:
            for i in range(10):
                print('nop')
        os_id_test.__init__(self, 0)
        self.seq_responses = responses
        self.seq_num_responses = len(responses)
        self.seq_base = seq_base

    def test_id(self):
        if False:
            for i in range(10):
                print('nop')
        return 'T1'

    def process(self):
        if False:
            print('Hello World!')
        if self.seq_num_responses < 1:
            self.add_result('R', 'N')
            return
        response = self.seq_responses[0]
        tests = nmap2_tcp_tests(response.get_ip(), response.get_tcp(), self.seq_base, nmap2_tcp_probe.acknowledgment)
        self.add_result('R', 'Y')
        self.add_result('DF', tests.get_df())
        self.add_result('S', tests.get_seq())
        self.add_result('A', tests.get_ack())
        self.add_result('F', tests.get_flags())
        self.add_result('Q', tests.get_quirks())

    def get_final_result(self):
        if False:
            while True:
                i = 10
        if not self.get_result_dict():
            return None
        else:
            return {self.test_id(): self.get_result_dict()}

class nmap2_icmp_container(os_id_test):

    def __init__(self, responses):
        if False:
            print('Hello World!')
        os_id_test.__init__(self, 0)
        self.icmp_responses = responses
        self.icmp_num_responses = len(responses)

    def test_id(self):
        if False:
            print('Hello World!')
        return 'IE'

    def process(self):
        if False:
            print('Hello World!')
        if self.icmp_num_responses != 2:
            self.add_result('R', 'N')
            return
        ip1 = self.icmp_responses[0].child()
        ip2 = self.icmp_responses[1].child()
        icmp1 = ip1.child()
        icmp2 = ip2.child()
        self.add_result('R', 'Y')
        if not ip1.get_ip_df() and (not ip2.get_ip_df()):
            self.add_result('DFI', 'N')
        elif ip1.get_ip_df() and (not ip2.get_ip_df()):
            self.add_result('DFI', 'S')
        elif ip1.get_ip_df() and ip2.get_ip_df():
            self.add_result('DFI', 'Y')
        else:
            self.add_result('DFI', 'O')
        if ip1.get_ip_tos() == 0 and ip2.get_ip_tos() == 0:
            self.add_result('TOSI', 'Z')
        elif ip1.get_ip_tos() == 0 and ip2.get_ip_tos() == 4:
            self.add_result('TOSI', 'S')
        elif ip1.get_ip_tos() == ip2.get_ip_tos():
            self.add_result('TOSI', '%X' % ip1.get_ip_tos())
        else:
            self.add_result('TOSI', 'O')
        if icmp1.get_icmp_code() == 0 and icmp2.get_icmp_code() == 0:
            self.add_result('CD', 'Z')
        elif icmp1.get_icmp_code() == 9 and icmp2.get_icmp_code() == 0:
            self.add_result('CD', 'S')
        elif icmp1.get_icmp_code() == icmp2.get_icmp_code():
            self.add_result('CD', '%X' % icmp1.get_icmp_code())
        else:
            self.add_result('CD', 'O')
        if icmp1.get_icmp_seq() == 0 and icmp2.get_icmp_seq() == 0:
            self.add_result('SI', 'Z')
        elif icmp1.get_icmp_seq() == nmap2_icmp_echo_probe_1.sequence_number and icmp2.get_icmp_seq() == nmap2_icmp_echo_probe_1.sequence_number + 1:
            self.add_result('SI', 'S')
        elif icmp1.get_icmp_seq() == icmp2.get_icmp_seq():
            self.add_result('SI', '%X' % icmp1.get_icmp_code())
        else:
            self.add_result('SI', 'O')

    def get_final_result(self):
        if False:
            print('Hello World!')
        if not self.get_result_dict():
            return None
        else:
            return {self.test_id(): self.get_result_dict()}

class nmap1_tcp_closed_1(nmap1_tcp_probe):

    def __init__(self, id, addresses, tcp_ports):
        if False:
            print('Hello World!')
        nmap1_tcp_probe.__init__(self, id, addresses, tcp_ports, 0)
        self.t.set_SYN()

    def test_id(self):
        if False:
            while True:
                i = 10
        return 'T5'

    def is_mine(self, packet):
        if False:
            while True:
                i = 10
        if tcp_probe.is_mine(self, packet):
            ip = packet.child()
            if not ip:
                return 0
            tcp = ip.child()
            if not tcp:
                return 0
            if tcp.get_RST():
                return 1
            else:
                return 0
        else:
            return 0

class nmap2_tcp_closed_1(nmap2_tcp_probe_2_6):

    def __init__(self, id, addresses, tcp_ports):
        if False:
            print('Hello World!')
        nmap2_tcp_probe_2_6.__init__(self, id, addresses, tcp_ports, 0)
        self.t.set_SYN()
        self.i.set_ip_df(0)
        self.t.set_th_win(31337)

    def test_id(self):
        if False:
            while True:
                i = 10
        return 'T5'

class nmap1_tcp_closed_2(nmap1_tcp_probe):

    def __init__(self, id, addresses, tcp_ports):
        if False:
            print('Hello World!')
        nmap1_tcp_probe.__init__(self, id, addresses, tcp_ports, 0)
        self.t.set_ACK()

    def test_id(self):
        if False:
            for i in range(10):
                print('nop')
        return 'T6'

class nmap2_tcp_closed_2(nmap2_tcp_probe_2_6):

    def __init__(self, id, addresses, tcp_ports):
        if False:
            return 10
        nmap2_tcp_probe_2_6.__init__(self, id, addresses, tcp_ports, 0)
        self.t.set_ACK()
        self.i.set_ip_df(1)
        self.t.set_th_win(32768)

    def test_id(self):
        if False:
            while True:
                i = 10
        return 'T6'

class nmap1_tcp_closed_3(nmap1_tcp_probe):

    def __init__(self, id, addresses, tcp_ports):
        if False:
            print('Hello World!')
        nmap1_tcp_probe.__init__(self, id, addresses, tcp_ports, 0)
        self.t.set_FIN()
        self.t.set_URG()
        self.t.set_PSH()

    def test_id(self):
        if False:
            for i in range(10):
                print('nop')
        return 'T7'

class nmap2_tcp_closed_3(nmap2_tcp_probe_7):

    def __init__(self, id, addresses, tcp_ports):
        if False:
            print('Hello World!')
        nmap2_tcp_probe_7.__init__(self, id, addresses, tcp_ports, 0)
        self.t.set_FIN()
        self.t.set_URG()
        self.t.set_PSH()
        self.t.set_th_win(65535)
        self.i.set_ip_df(0)

    def test_id(self):
        if False:
            for i in range(10):
                print('nop')
        return 'T7'

class NMAP2_OS_Class:

    def __init__(self, vendor, name, family, device_type):
        if False:
            i = 10
            return i + 15
        self.__vendor = vendor
        self.__name = name
        self.__family = family
        self.__device_type = device_type

    def get_vendor(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__vendor

    def get_name(self):
        if False:
            i = 10
            return i + 15
        return self.__name

    def get_family(self):
        if False:
            while True:
                i = 10
        return self.__family

    def get_device_type(self):
        if False:
            print('Hello World!')
        return self.__device_type

class NMAP2_Fingerprint:

    def __init__(self, id, os_class, tests):
        if False:
            i = 10
            return i + 15
        self.__id = id
        self.__os_class = os_class
        self.__tests = tests

    def get_id(self):
        if False:
            i = 10
            return i + 15
        return self.__id

    def get_os_class(self):
        if False:
            print('Hello World!')
        return self.__os_class

    def get_tests(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__tests

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        ret = 'FP: [%s]' % self.__id
        ret += '\n vendor: %s' % self.__os_class.get_vendor()
        ret += '\n name: %s' % self.__os_class.get_name()
        ret += '\n family: %s' % self.__os_class.get_family()
        ret += '\n device_type: %s' % self.__os_class.get_device_type()
        for test in self.__tests:
            ret += '\n  test: %s' % test
            for pair in self.__tests[test]:
                ret += '\n   %s = [%s]' % (pair, self.__tests[test][pair])
        return ret
    literal_conv = {'RIPL': {'G': 328}, 'RID': {'G': 4162}, 'RUL': {'G': 308}}

    def parse_int(self, field, value):
        if False:
            print('Hello World!')
        try:
            return int(value, 16)
        except ValueError:
            if field in NMAP2_Fingerprint.literal_conv:
                if value in NMAP2_Fingerprint.literal_conv[field]:
                    return NMAP2_Fingerprint.literal_conv[field][value]
            return 0

    def match(self, field, ref, value):
        if False:
            return 10
        options = ref.split('|')
        for option in options:
            if option.startswith('>'):
                if self.parse_int(field, value) > self.parse_int(field, option[1:]):
                    return True
            elif option.startswith('<'):
                if self.parse_int(field, value) < self.parse_int(field, option[1:]):
                    return True
            elif option.find('-') > -1:
                range = option.split('-')
                if self.parse_int(field, value) >= self.parse_int(field, range[0]) and self.parse_int(field, value) <= self.parse_int(field, range[1]):
                    return True
            elif str(value) == str(option):
                return True
        return False

    def compare(self, sample, mp):
        if False:
            return 10
        max_points = 0
        total_points = 0
        for test in self.__tests:
            if test not in sample:
                continue
            for field in self.__tests[test]:
                if field not in sample[test] or test not in mp or field not in mp[test]:
                    continue
                ref = self.__tests[test][field]
                value = sample[test][field]
                points = int(mp[test][field])
                max_points += points
                if self.match(field, ref, value):
                    total_points += points
        return total_points / float(max_points) * 100

class NMAP2_Fingerprint_Matcher:

    def __init__(self, filename):
        if False:
            while True:
                i = 10
        self.__filename = filename

    def find_matches(self, res, threshold):
        if False:
            while True:
                i = 10
        output = []
        try:
            infile = open(self.__filename, 'r')
            mp = self.parse_mp(self.matchpoints(infile))
            for fingerprint in self.fingerprints(infile):
                fp = self.parse_fp(fingerprint)
                similarity = fp.compare(res, mp)
                if similarity >= threshold:
                    print('"%s" matches with an accuracy of %.2f%%' % (fp.get_id(), similarity))
                    output.append((similarity / 100, fp.get_id(), (fp.get_os_class().get_vendor(), fp.get_os_class().get_name(), fp.get_os_class().get_family(), fp.get_os_class().get_device_type())))
            infile.close()
        except IOError as err:
            print('IOError: %s', err)
        return output

    def sections(self, infile, token):
        if False:
            i = 10
            return i + 15
        OUT = 0
        IN = 1
        state = OUT
        output = []
        for line in infile:
            line = line.strip()
            if state == OUT:
                if line.startswith(token):
                    state = IN
                    output = [line]
            elif state == IN:
                if line:
                    output.append(line)
                else:
                    state = OUT
                    yield output
                    output = []
        if output:
            yield output

    def fingerprints(self, infile):
        if False:
            i = 10
            return i + 15
        for section in self.sections(infile, 'Fingerprint'):
            yield section

    def matchpoints(self, infile):
        if False:
            print('Hello World!')
        return self.sections(infile, 'MatchPoints').next()

    def parse_line(self, line):
        if False:
            return 10
        name = line[:line.find('(')]
        pairs = line[line.find('(') + 1:line.find(')')]
        test = {}
        for pair in pairs.split('%'):
            pair = pair.split('=')
            test[pair[0]] = pair[1]
        return (name, test)

    def parse_fp(self, fp):
        if False:
            return 10
        tests = {}
        for line in fp:
            if line.startswith('#'):
                continue
            elif line.startswith('Fingerprint'):
                fingerprint = line[len('Fingerprint') + 1:]
            elif line.startswith('Class'):
                (vendor, name, family, device_type) = line[len('Class') + 1:].split('|')
                os_class = NMAP2_OS_Class(vendor.strip(), name.strip(), family.strip(), device_type.strip())
            else:
                test = self.parse_line(line)
                tests[test[0]] = test[1]
        return NMAP2_Fingerprint(fingerprint, os_class, tests)

    def parse_mp(self, fp):
        if False:
            print('Hello World!')
        tests = {}
        for line in fp:
            if line.startswith('#'):
                continue
            elif line.startswith('MatchPoints'):
                continue
            else:
                test = self.parse_line(line)
                tests[test[0]] = test[1]
        return tests