"""
Clone of p0f v2 passive OS fingerprinting
"""
import time
import struct
import os
import socket
import random
from scapy.data import KnowledgeBase, select_path
from scapy.config import conf
from scapy.compat import raw
from scapy.layers.inet import IP, TCP, TCPOptions
from scapy.packet import NoPayload, Packet
from scapy.error import warning, Scapy_Exception, log_runtime
from scapy.volatile import RandInt, RandByte, RandNum, RandShort, RandString
from scapy.sendrecv import sniff
if conf.route is None:
    import scapy.route
_p0fpaths = ['/etc/p0f', '/usr/share/p0f', '/opt/local']
conf.p0f_base = select_path(_p0fpaths, 'p0f.fp')
conf.p0fa_base = select_path(_p0fpaths, 'p0fa.fp')
conf.p0fr_base = select_path(_p0fpaths, 'p0fr.fp')
conf.p0fo_base = select_path(_p0fpaths, 'p0fo.fp')

class p0fKnowledgeBase(KnowledgeBase):

    def __init__(self, filename):
        if False:
            for i in range(10):
                print('nop')
        KnowledgeBase.__init__(self, filename)

    def lazy_init(self):
        if False:
            return 10
        try:
            f = open(self.filename)
        except IOError:
            warning("Can't open base %s", self.filename)
            return
        try:
            self.base = []
            for line in f:
                if line[0] in ['#', '\n']:
                    continue
                line = tuple(line.split(':'))
                if len(line) < 8:
                    continue

                def a2i(x):
                    if False:
                        i = 10
                        return i + 15
                    if x.isdigit():
                        return int(x)
                    return x
                li = [a2i(e) for e in line[1:4]]
                self.base.append((line[0], li[0], li[1], li[2], line[4], line[5], line[6], line[7][:-1]))
        except Exception:
            warning("Can't parse p0f database (new p0f version ?)")
            self.base = None
        f.close()
(p0f_kdb, p0fa_kdb, p0fr_kdb, p0fo_kdb) = (None, None, None, None)

def p0f_load_knowledgebases():
    if False:
        for i in range(10):
            print('nop')
    global p0f_kdb, p0fa_kdb, p0fr_kdb, p0fo_kdb
    p0f_kdb = p0fKnowledgeBase(conf.p0f_base)
    p0fa_kdb = p0fKnowledgeBase(conf.p0fa_base)
    p0fr_kdb = p0fKnowledgeBase(conf.p0fr_base)
    p0fo_kdb = p0fKnowledgeBase(conf.p0fo_base)
p0f_load_knowledgebases()

def p0f_selectdb(flags):
    if False:
        while True:
            i = 10
    if flags & 22 == 2:
        return p0f_kdb
    elif flags & 22 == 18:
        return p0fa_kdb
    elif flags & 22 in [4, 20]:
        return p0fr_kdb
    elif flags & 22 == 16:
        return p0fo_kdb
    else:
        return None

def packet2p0f(pkt):
    if False:
        for i in range(10):
            print('nop')
    pkt = pkt.copy()
    pkt = pkt.__class__(raw(pkt))
    while pkt.haslayer(IP) and pkt.haslayer(TCP):
        pkt = pkt.getlayer(IP)
        if isinstance(pkt.payload, TCP):
            break
        pkt = pkt.payload
    if not isinstance(pkt, IP) or not isinstance(pkt.payload, TCP):
        raise TypeError('Not a TCP/IP packet')
    db = p0f_selectdb(pkt.payload.flags)
    ttl = pkt.ttl
    ss = len(pkt)
    if ss > 100:
        if db == p0fr_kdb:
            ss = '*'
        else:
            ss = 0
    if db == p0fo_kdb:
        ss = '*'
    ooo = ''
    mss = -1
    qqT = False
    qqP = False
    ilen = (pkt.payload.dataofs << 2) - 20
    for option in pkt.payload.options:
        ilen -= 1
        if option[0] == 'MSS':
            ooo += 'M' + str(option[1]) + ','
            mss = option[1]
            ilen -= 3
        elif option[0] == 'WScale':
            ooo += 'W' + str(option[1]) + ','
            ilen -= 2
        elif option[0] == 'Timestamp':
            if option[1][0] == 0:
                ooo += 'T0,'
            else:
                ooo += 'T,'
            if option[1][1] != 0:
                qqT = True
            ilen -= 9
        elif option[0] == 'SAckOK':
            ooo += 'S,'
            ilen -= 1
        elif option[0] == 'NOP':
            ooo += 'N,'
        elif option[0] == 'EOL':
            ooo += 'E,'
            if ilen > 0:
                qqP = True
        elif isinstance(option[0], str):
            ooo += '?%i,' % TCPOptions[1][option[0]]
        else:
            ooo += '?%i,' % option[0]
    ooo = ooo[:-1]
    if ooo == '':
        ooo = '.'
    win = pkt.payload.window
    if mss != -1:
        if mss != 0 and win % mss == 0:
            win = 'S' + str(win / mss)
        elif win % (mss + 40) == 0:
            win = 'T' + str(win / (mss + 40))
    win = str(win)
    qq = ''
    if db == p0fr_kdb:
        if pkt.payload.flags & 16 == 16:
            qq += 'K'
    if pkt.payload.seq == pkt.payload.ack:
        qq += 'Q'
    if pkt.payload.seq == 0:
        qq += '0'
    if qqP:
        qq += 'P'
    if pkt.id == 0:
        qq += 'Z'
    if pkt.options != []:
        qq += 'I'
    if pkt.payload.urgptr != 0:
        qq += 'U'
    if pkt.payload.reserved != 0:
        qq += 'X'
    if pkt.payload.ack != 0:
        qq += 'A'
    if qqT:
        qq += 'T'
    if db == p0fo_kdb:
        if pkt.payload.flags & 32 != 0:
            qq += 'F'
    elif pkt.payload.flags & 40 != 0:
        qq += 'F'
    if db != p0fo_kdb and (not isinstance(pkt.payload.payload, NoPayload)):
        qq += 'D'
    if qq == '':
        qq = '.'
    return (db, (win, ttl, pkt.flags.DF, ss, ooo, qq))

def p0f_correl(x, y):
    if False:
        for i in range(10):
            print('nop')
    d = 0
    d += x[0] == y[0] or y[0] == '*' or (y[0][0] == '%' and x[0].isdigit() and (int(x[0]) % int(y[0][1:]) == 0))
    d += y[1] >= x[1] and y[1] - x[1] < 32
    for i in [2, 5]:
        d += x[i] == y[i] or y[i] == '*'
    d += x[3] == y[3]
    xopt = x[4].split(',')
    yopt = y[4].split(',')
    if len(xopt) == len(yopt):
        same = True
        for i in range(len(xopt)):
            if not (xopt[i] == yopt[i] or (len(yopt[i]) == 2 and len(xopt[i]) > 1 and (yopt[i][1] == '*') and (xopt[i][0] == yopt[i][0])) or (len(yopt[i]) > 2 and len(xopt[i]) > 1 and (yopt[i][1] == '%') and (xopt[i][0] == yopt[i][0]) and (int(xopt[i][1:]) % int(yopt[i][2:]) == 0))):
                same = False
                break
        if same:
            d += len(xopt)
    return d

@conf.commands.register
def p0f(pkt):
    if False:
        return 10
    'Passive OS fingerprinting: which OS emitted this TCP packet ?\np0f(packet) -> accuracy, [list of guesses]\n'
    (db, sig) = packet2p0f(pkt)
    if db:
        pb = db.get_base()
    else:
        pb = []
    if not pb:
        warning('p0f base empty.')
        return []
    r = []
    max = len(sig[4].split(',')) + 5
    for b in pb:
        d = p0f_correl(sig, b)
        if d == max:
            r.append((b[6], b[7], b[1] - pkt[IP].ttl))
    return r

def prnp0f(pkt):
    if False:
        for i in range(10):
            print('nop')
    'Calls p0f and returns a user-friendly output'
    try:
        r = p0f(pkt)
    except Exception:
        return
    if r == []:
        r = ('UNKNOWN', '[' + ':'.join(map(str, packet2p0f(pkt)[1])) + ':?:?]', None)
    else:
        r = r[0]
    uptime = None
    try:
        uptime = pkt2uptime(pkt)
    except Exception:
        pass
    if uptime == 0:
        uptime = None
    res = pkt.sprintf('%IP.src%:%TCP.sport% - ' + r[0] + ' ' + r[1])
    if uptime is not None:
        res += pkt.sprintf(' (up: ' + str(uptime / 3600) + ' hrs)\n  -> %IP.dst%:%TCP.dport% (%TCP.flags%)')
    else:
        res += pkt.sprintf('\n  -> %IP.dst%:%TCP.dport% (%TCP.flags%)')
    if r[2] is not None:
        res += ' (distance ' + str(r[2]) + ')'
    print(res)

@conf.commands.register
def pkt2uptime(pkt, HZ=100):
    if False:
        for i in range(10):
            print('nop')
    'Calculate the date the machine which emitted the packet booted using TCP timestamp  # noqa: E501\npkt2uptime(pkt, [HZ=100])'
    if not isinstance(pkt, Packet):
        raise TypeError('Not a TCP packet')
    if isinstance(pkt, NoPayload):
        raise TypeError('Not a TCP packet')
    if not isinstance(pkt, TCP):
        return pkt2uptime(pkt.payload)
    for opt in pkt.options:
        if opt[0] == 'Timestamp':
            t = opt[1][0] / HZ
            return t
    raise TypeError('No timestamp option')

def p0f_impersonate(pkt, osgenre=None, osdetails=None, signature=None, extrahops=0, mtu=1500, uptime=None):
    if False:
        return 10
    'Modifies pkt so that p0f will think it has been sent by a\nspecific OS.  If osdetails is None, then we randomly pick up a\npersonality matching osgenre. If osgenre and signature are also None,\nwe use a local signature (using p0f_getlocalsigs). If signature is\nspecified (as a tuple), we use the signature.\n\nFor now, only TCP Syn packets are supported.\nSome specifications of the p0f.fp file are not (yet) implemented.'
    pkt = pkt.copy()
    while pkt.haslayer(IP) and pkt.haslayer(TCP):
        pkt = pkt.getlayer(IP)
        if isinstance(pkt.payload, TCP):
            break
        pkt = pkt.payload
    if not isinstance(pkt, IP) or not isinstance(pkt.payload, TCP):
        raise TypeError('Not a TCP/IP packet')
    db = p0f_selectdb(pkt.payload.flags)
    if osgenre:
        pb = db.get_base()
        if pb is None:
            pb = []
        pb = [x for x in pb if x[6] == osgenre]
        if osdetails:
            pb = [x for x in pb if x[7] == osdetails]
    elif signature:
        pb = [signature]
    else:
        pb = p0f_getlocalsigs()[db]
    if db == p0fr_kdb:
        if pkt.payload.flags & 4 == 4:
            pb = [x for x in pb if 'K' in x[5]]
        else:
            pb = [x for x in pb if 'K' not in x[5]]
    if not pb:
        raise Scapy_Exception('No match in the p0f database')
    pers = pb[random.randint(0, len(pb) - 1)]
    orig_opts = dict(pkt.payload.options)
    int_only = lambda val: val if isinstance(val, int) else None
    mss_hint = int_only(orig_opts.get('MSS'))
    wscale_hint = int_only(orig_opts.get('WScale'))
    ts_hint = [int_only(o) for o in orig_opts.get('Timestamp', (None, None))]
    options = []
    if pers[4] != '.':
        for opt in pers[4].split(','):
            if opt[0] == 'M':
                if pers[0][0] == 'S':
                    maxmss = (2 ** 16 - 1) // int(pers[0][1:])
                else:
                    maxmss = 2 ** 16 - 1
                if mss_hint and (not 0 <= mss_hint <= maxmss):
                    mss_hint = None
                if opt[1:] == '*':
                    if mss_hint is not None:
                        options.append(('MSS', mss_hint))
                    else:
                        options.append(('MSS', random.randint(1, maxmss)))
                elif opt[1] == '%':
                    coef = int(opt[2:])
                    if mss_hint is not None and mss_hint % coef == 0:
                        options.append(('MSS', mss_hint))
                    else:
                        options.append(('MSS', coef * random.randint(1, maxmss // coef)))
                else:
                    options.append(('MSS', int(opt[1:])))
            elif opt[0] == 'W':
                if wscale_hint and (not 0 <= wscale_hint < 2 ** 8):
                    wscale_hint = None
                if opt[1:] == '*':
                    if wscale_hint is not None:
                        options.append(('WScale', wscale_hint))
                    else:
                        options.append(('WScale', RandByte()))
                elif opt[1] == '%':
                    coef = int(opt[2:])
                    if wscale_hint is not None and wscale_hint % coef == 0:
                        options.append(('WScale', wscale_hint))
                    else:
                        options.append(('WScale', coef * RandNum(min=1, max=(2 ** 8 - 1) // coef)))
                else:
                    options.append(('WScale', int(opt[1:])))
            elif opt == 'T0':
                options.append(('Timestamp', (0, 0)))
            elif opt == 'T':
                if uptime is not None:
                    ts_a = uptime
                elif ts_hint[0] and 0 < ts_hint[0] < 2 ** 32:
                    ts_a = ts_hint[0]
                else:
                    ts_a = random.randint(120, 100 * 60 * 60 * 24 * 365)
                if 'T' not in pers[5]:
                    ts_b = 0
                elif ts_hint[1] and 0 < ts_hint[1] < 2 ** 32:
                    ts_b = ts_hint[1]
                else:
                    ts_b = random.randint(1, 2 ** 32 - 1)
                options.append(('Timestamp', (ts_a, ts_b)))
            elif opt == 'S':
                options.append(('SAckOK', ''))
            elif opt == 'N':
                options.append(('NOP', None))
            elif opt == 'E':
                options.append(('EOL', None))
            elif opt[0] == '?':
                if int(opt[1:]) in TCPOptions[0]:
                    optname = TCPOptions[0][int(opt[1:])][0]
                    optstruct = TCPOptions[0][int(opt[1:])][1]
                    options.append((optname, struct.unpack(optstruct, RandString(struct.calcsize(optstruct))._fix())))
                else:
                    options.append((int(opt[1:]), ''))
            else:
                warning('unhandled TCP option %s', opt)
            pkt.payload.options = options
    if pers[0] == '*':
        pkt.payload.window = RandShort()
    elif pers[0].isdigit():
        pkt.payload.window = int(pers[0])
    elif pers[0][0] == '%':
        coef = int(pers[0][1:])
        pkt.payload.window = coef * RandNum(min=1, max=(2 ** 16 - 1) // coef)
    elif pers[0][0] == 'T':
        pkt.payload.window = mtu * int(pers[0][1:])
    elif pers[0][0] == 'S':
        mss = [x for x in options if x[0] == 'MSS']
        if not mss:
            raise Scapy_Exception('TCP window value requires MSS, and MSS option not set')
        pkt.payload.window = mss[0][1] * int(pers[0][1:])
    else:
        raise Scapy_Exception('Unhandled window size specification')
    pkt.ttl = pers[1] - extrahops
    pkt.flags |= 2 * pers[2]
    if pers[5] != '.':
        for qq in pers[5]:
            if qq == 'Z':
                pkt.id = 0
            elif qq == 'U':
                pkt.payload.urgptr = RandShort()
            elif qq == 'A':
                pkt.payload.ack = RandInt()
            elif qq == 'F':
                if db == p0fo_kdb:
                    pkt.payload.flags |= 32
                else:
                    pkt.payload.flags |= random.choice([8, 32, 40])
            elif qq == 'D' and db != p0fo_kdb:
                pkt /= conf.raw_layer(load=RandString(random.randint(1, 10)))
            elif qq == 'Q':
                pkt.payload.seq = pkt.payload.ack
    if '0' in pers[5]:
        pkt.payload.seq = 0
    elif pkt.payload.seq == 0:
        pkt.payload.seq = RandInt()
    while pkt.underlayer:
        pkt = pkt.underlayer
    return pkt

def p0f_getlocalsigs():
    if False:
        for i in range(10):
            print('nop')
    'This function returns a dictionary of signatures indexed by p0f\ndb (e.g., p0f_kdb, p0fa_kdb, ...) for the local TCP/IP stack.\n\nYou need to have your firewall at least accepting the TCP packets\nfrom/to a high port (30000 <= x <= 40000) on your loopback interface.\n\nPlease note that the generated signatures come from the loopback\ninterface and may (are likely to) be different than those generated on\n"normal" interfaces.'
    pid = os.fork()
    port = random.randint(30000, 40000)
    if pid > 0:
        result = {}

        def addresult(res):
            if False:
                print('Hello World!')
            if res[0] not in result:
                result[res[0]] = [res[1]]
            elif res[1] not in result[res[0]]:
                result[res[0]].append(res[1])
        iface = conf.route.route('127.0.0.1')[0]
        count = 14
        pl = sniff(iface=iface, filter='tcp and port ' + str(port), count=count, timeout=3)
        for pkt in pl:
            for elt in packet2p0f(pkt):
                addresult(elt)
        os.waitpid(pid, 0)
    elif pid < 0:
        log_runtime.error('fork error')
    else:
        time.sleep(1)
        s1 = socket.socket(socket.AF_INET, type=socket.SOCK_STREAM)
        try:
            s1.connect(('127.0.0.1', port))
        except socket.error:
            pass
        s1.bind(('127.0.0.1', port))
        s1.connect(('127.0.0.1', port))
        s1.close()
        os._exit(0)
    return result