import threading, random, operator, time
import SocketServer, socket, sys, os
import binascii
import string
import base64
import time
import logging
from configobj import ConfigObj
from core.configwatcher import ConfigWatcher
from core.utils import shutdown
from core.logger import logger
from dnslib import *
from IPy import IP
formatter = logging.Formatter('%(asctime)s %(clientip)s [DNS] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logger().setup_logger('DNSChef', formatter)
dnslog = logging.getLogger('dnslog')
handler = logging.FileHandler('./logs/dns/dns.log')
handler.setFormatter(formatter)
dnslog.addHandler(handler)
dnslog.setLevel(logging.INFO)

class DNSHandler:

    def parse(self, data):
        if False:
            print('Hello World!')
        nametodns = DNSChef().nametodns
        nameservers = DNSChef().nameservers
        hsts = DNSChef().hsts
        hstsconfig = DNSChef().real_records
        server_address = DNSChef().server_address
        clientip = {'clientip': self.client_address[0]}
        response = ''
        try:
            d = DNSRecord.parse(data)
        except Exception as e:
            log.info('Error: invalid DNS request', extra=clientip)
            dnslog.info('Error: invalid DNS request', extra=clientip)
        else:
            if QR[d.header.qr] == 'QUERY':
                qname = str(d.q.qname)
                if qname[-1] == '.':
                    qname = qname[:-1]
                qtype = QTYPE[d.q.qtype]
                fake_records = dict()
                for record in nametodns:
                    fake_records[record] = self.findnametodns(qname, nametodns[record])
                if hsts:
                    if qname in hstsconfig:
                        response = self.hstsbypass(hstsconfig[qname], qname, nameservers, d)
                        return response
                    elif qname[:4] == 'wwww':
                        response = self.hstsbypass(qname[1:], qname, nameservers, d)
                        return response
                    elif qname[:3] == 'web':
                        response = self.hstsbypass(qname[3:], qname, nameservers, d)
                        return response
                if qtype in fake_records and fake_records[qtype]:
                    fake_record = fake_records[qtype]
                    response = DNSRecord(DNSHeader(id=d.header.id, bitmap=d.header.bitmap, qr=1, aa=1, ra=1), q=d.q)
                    log.info("Cooking the response of type '{}' for {} to {}".format(qtype, qname, fake_record), extra=clientip)
                    dnslog.info("Cooking the response of type '{}' for {} to {}".format(qtype, qname, fake_record), extra=clientip)
                    if qtype == 'AAAA':
                        ipv6 = IP(fake_record)
                        ipv6_bin = ipv6.strBin()
                        ipv6_hex_tuple = [int(ipv6_bin[i:i + 8], 2) for i in xrange(0, len(ipv6_bin), 8)]
                        response.add_answer(RR(qname, getattr(QTYPE, qtype), rdata=RDMAP[qtype](ipv6_hex_tuple)))
                    elif qtype == 'SOA':
                        (mname, rname, t1, t2, t3, t4, t5) = fake_record.split(' ')
                        times = tuple([int(t) for t in [t1, t2, t3, t4, t5]])
                        if mname[-1] == '.':
                            mname = mname[:-1]
                        if rname[-1] == '.':
                            rname = rname[:-1]
                        response.add_answer(RR(qname, getattr(QTYPE, qtype), rdata=RDMAP[qtype](mname, rname, times)))
                    elif qtype == 'NAPTR':
                        (order, preference, flags, service, regexp, replacement) = fake_record.split(' ')
                        order = int(order)
                        preference = int(preference)
                        if replacement[-1] == '.':
                            replacement = replacement[:-1]
                        response.add_answer(RR(qname, getattr(QTYPE, qtype), rdata=RDMAP[qtype](order, preference, flags, service, regexp, DNSLabel(replacement))))
                    elif qtype == 'SRV':
                        (priority, weight, port, target) = fake_record.split(' ')
                        priority = int(priority)
                        weight = int(weight)
                        port = int(port)
                        if target[-1] == '.':
                            target = target[:-1]
                        response.add_answer(RR(qname, getattr(QTYPE, qtype), rdata=RDMAP[qtype](priority, weight, port, target)))
                    elif qtype == 'DNSKEY':
                        (flags, protocol, algorithm, key) = fake_record.split(' ')
                        flags = int(flags)
                        protocol = int(protocol)
                        algorithm = int(algorithm)
                        key = base64.b64decode(''.join(key).encode('ascii'))
                        response.add_answer(RR(qname, getattr(QTYPE, qtype), rdata=RDMAP[qtype](flags, protocol, algorithm, key)))
                    elif qtype == 'RRSIG':
                        (covered, algorithm, labels, orig_ttl, sig_exp, sig_inc, key_tag, name, sig) = fake_record.split(' ')
                        covered = getattr(QTYPE, covered)
                        algorithm = int(algorithm)
                        labels = int(labels)
                        orig_ttl = int(orig_ttl)
                        sig_exp = int(time.mktime(time.strptime(sig_exp + 'GMT', '%Y%m%d%H%M%S%Z')))
                        sig_inc = int(time.mktime(time.strptime(sig_inc + 'GMT', '%Y%m%d%H%M%S%Z')))
                        key_tag = int(key_tag)
                        if name[-1] == '.':
                            name = name[:-1]
                        sig = base64.b64decode(''.join(sig).encode('ascii'))
                        response.add_answer(RR(qname, getattr(QTYPE, qtype), rdata=RDMAP[qtype](covered, algorithm, labels, orig_ttl, sig_exp, sig_inc, key_tag, name, sig)))
                    else:
                        if fake_record[-1] == '.':
                            fake_record = fake_record[:-1]
                        response.add_answer(RR(qname, getattr(QTYPE, qtype), rdata=RDMAP[qtype](fake_record)))
                    response = response.pack()
                elif qtype == '*' and (not None in fake_records.values()):
                    log.info("Cooking the response of type '{}' for {} with {}".format('ANY', qname, 'all known fake records.'), extra=clientip)
                    dnslog.info("Cooking the response of type '{}' for {} with {}".format('ANY', qname, 'all known fake records.'), extra=clientip)
                    response = DNSRecord(DNSHeader(id=d.header.id, bitmap=d.header.bitmap, qr=1, aa=1, ra=1), q=d.q)
                    for (qtype, fake_record) in fake_records.items():
                        if fake_record:
                            if qtype == 'AAAA':
                                ipv6 = IP(fake_record)
                                ipv6_bin = ipv6.strBin()
                                fake_record = [int(ipv6_bin[i:i + 8], 2) for i in xrange(0, len(ipv6_bin), 8)]
                            elif qtype == 'SOA':
                                (mname, rname, t1, t2, t3, t4, t5) = fake_record.split(' ')
                                times = tuple([int(t) for t in [t1, t2, t3, t4, t5]])
                                if mname[-1] == '.':
                                    mname = mname[:-1]
                                if rname[-1] == '.':
                                    rname = rname[:-1]
                                response.add_answer(RR(qname, getattr(QTYPE, qtype), rdata=RDMAP[qtype](mname, rname, times)))
                            elif qtype == 'NAPTR':
                                (order, preference, flags, service, regexp, replacement) = fake_record.split(' ')
                                order = int(order)
                                preference = int(preference)
                                if replacement and replacement[-1] == '.':
                                    replacement = replacement[:-1]
                                response.add_answer(RR(qname, getattr(QTYPE, qtype), rdata=RDMAP[qtype](order, preference, flags, service, regexp, replacement)))
                            elif qtype == 'SRV':
                                (priority, weight, port, target) = fake_record.split(' ')
                                priority = int(priority)
                                weight = int(weight)
                                port = int(port)
                                if target[-1] == '.':
                                    target = target[:-1]
                                response.add_answer(RR(qname, getattr(QTYPE, qtype), rdata=RDMAP[qtype](priority, weight, port, target)))
                            elif qtype == 'DNSKEY':
                                (flags, protocol, algorithm, key) = fake_record.split(' ')
                                flags = int(flags)
                                protocol = int(protocol)
                                algorithm = int(algorithm)
                                key = base64.b64decode(''.join(key).encode('ascii'))
                                response.add_answer(RR(qname, getattr(QTYPE, qtype), rdata=RDMAP[qtype](flags, protocol, algorithm, key)))
                            elif qtype == 'RRSIG':
                                (covered, algorithm, labels, orig_ttl, sig_exp, sig_inc, key_tag, name, sig) = fake_record.split(' ')
                                covered = getattr(QTYPE, covered)
                                algorithm = int(algorithm)
                                labels = int(labels)
                                orig_ttl = int(orig_ttl)
                                sig_exp = int(time.mktime(time.strptime(sig_exp + 'GMT', '%Y%m%d%H%M%S%Z')))
                                sig_inc = int(time.mktime(time.strptime(sig_inc + 'GMT', '%Y%m%d%H%M%S%Z')))
                                key_tag = int(key_tag)
                                if name[-1] == '.':
                                    name = name[:-1]
                                sig = base64.b64decode(''.join(sig).encode('ascii'))
                                response.add_answer(RR(qname, getattr(QTYPE, qtype), rdata=RDMAP[qtype](covered, algorithm, labels, orig_ttl, sig_exp, sig_inc, key_tag, name, sig)))
                            else:
                                if fake_record[-1] == '.':
                                    fake_record = fake_record[:-1]
                                response.add_answer(RR(qname, getattr(QTYPE, qtype), rdata=RDMAP[qtype](fake_record)))
                    response = response.pack()
                else:
                    log.debug("Proxying the response of type '{}' for {}".format(qtype, qname), extra=clientip)
                    dnslog.info("Proxying the response of type '{}' for {}".format(qtype, qname), extra=clientip)
                    nameserver_tuple = random.choice(nameservers).split('#')
                    response = self.proxyrequest(data, *nameserver_tuple)
        return response

    def findnametodns(self, qname, nametodns):
        if False:
            for i in range(10):
                print('nop')
        qname = qname.lower()
        qnamelist = qname.split('.')
        qnamelist.reverse()
        for (domain, host) in sorted(nametodns.iteritems(), key=operator.itemgetter(1)):
            domain = domain.split('.')
            domain.reverse()
            for (a, b) in map(None, qnamelist, domain):
                if a != b and b != '*':
                    break
            else:
                return host
        else:
            return False

    def proxyrequest(self, request, host, port='53', protocol='udp'):
        if False:
            print('Hello World!')
        clientip = {'clientip': self.client_address[0]}
        reply = None
        try:
            if DNSChef().ipv6:
                if protocol == 'udp':
                    sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
                elif protocol == 'tcp':
                    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            elif protocol == 'udp':
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            elif protocol == 'tcp':
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            if protocol == 'udp':
                sock.sendto(request, (host, int(port)))
                reply = sock.recv(1024)
                sock.close()
            elif protocol == 'tcp':
                sock.connect((host, int(port)))
                length = binascii.unhexlify('%04x' % len(request))
                sock.sendall(length + request)
                reply = sock.recv(1024)
                reply = reply[2:]
                sock.close()
        except Exception as e:
            log.warning('Could not proxy request: {}'.format(e), extra=clientip)
            dnslog.info('Could not proxy request: {}'.format(e), extra=clientip)
        else:
            return reply

    def hstsbypass(self, real_domain, fake_domain, nameservers, d):
        if False:
            i = 10
            return i + 15
        clientip = {'clientip': self.client_address[0]}
        log.info("Resolving '{}' to '{}' for HSTS bypass".format(fake_domain, real_domain), extra=clientip)
        dnslog.info("Resolving '{}' to '{}' for HSTS bypass".format(fake_domain, real_domain), extra=clientip)
        response = DNSRecord(DNSHeader(id=d.header.id, bitmap=d.header.bitmap, qr=1, aa=1, ra=1), q=d.q)
        nameserver_tuple = random.choice(nameservers).split('#')
        q = DNSRecord.question(real_domain).pack()
        r = self.proxyrequest(q, *nameserver_tuple)
        if r is None:
            return None
        dns_rr = DNSRecord.parse(r).rr
        for res in dns_rr:
            if res.get_rname() == real_domain:
                res.set_rname(fake_domain)
                response.add_answer(res)
            else:
                response.add_answer(res)
        return response.pack()

class UDPHandler(DNSHandler, SocketServer.BaseRequestHandler):

    def handle(self):
        if False:
            i = 10
            return i + 15
        (data, socket) = self.request
        response = self.parse(data)
        if response:
            socket.sendto(response, self.client_address)

class TCPHandler(DNSHandler, SocketServer.BaseRequestHandler):

    def handle(self):
        if False:
            print('Hello World!')
        data = self.request.recv(1024)
        data = data[2:]
        response = self.parse(data)
        if response:
            length = binascii.unhexlify('%04x' % len(response))
            self.request.sendall(length + response)

class ThreadedUDPServer(SocketServer.ThreadingMixIn, SocketServer.UDPServer):

    def __init__(self, server_address, RequestHandlerClass):
        if False:
            while True:
                i = 10
        self.address_family = socket.AF_INET6 if DNSChef().ipv6 else socket.AF_INET
        SocketServer.UDPServer.__init__(self, server_address, RequestHandlerClass)

class ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass):
        if False:
            while True:
                i = 10
        self.address_family = socket.AF_INET6 if DNSChef().ipv6 else socket.AF_INET
        SocketServer.TCPServer.__init__(self, server_address, RequestHandlerClass)

class DNSChef(ConfigWatcher):
    version = '0.4'
    tcp = False
    ipv6 = False
    hsts = False
    real_records = {}
    nametodns = {}
    server_address = '0.0.0.0'
    nameservers = ['8.8.8.8']
    port = 53
    __shared_state = {}

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__dict__ = self.__shared_state

    def on_config_change(self):
        if False:
            print('Hello World!')
        config = self.config['MITMf']['DNS']
        self.port = int(config['port'])
        for qtype in RDMAP.keys():
            self.nametodns[qtype] = dict()
        if config['ipv6'].lower() == 'on':
            self.ipv6 = True
            if config['nameservers'] == '8.8.8.8':
                self.nameservers = '2001:4860:4860::8888'
        if config['nameservers']:
            self.nameservers = []
            if type(config['nameservers']) is str:
                self.nameservers.append(config['nameservers'])
            elif type(config['nameservers']) is list:
                self.nameservers = config['nameservers']
        for section in config.sections:
            if section in self.nametodns:
                for (domain, record) in config[section].iteritems():
                    domain = domain.lower()
                    self.nametodns[section][domain] = record
        for (k, v) in self.config['SSLstrip+'].iteritems():
            self.real_records[v] = k

    def setHstsBypass(self):
        if False:
            while True:
                i = 10
        self.hsts = True

    def start(self):
        if False:
            return 10
        self.on_config_change()
        self.start_config_watch()
        try:
            if self.config['MITMf']['DNS']['tcp'].lower() == 'on':
                self.startTCP()
            else:
                self.startUDP()
        except socket.error as e:
            if 'Address already in use' in e:
                shutdown('\n[DNS] Unable to start DNS server on port {}: port already in use'.format(self.config['MITMf']['DNS']['port']))

    def startUDP(self):
        if False:
            print('Hello World!')
        server = ThreadedUDPServer((self.server_address, int(self.port)), UDPHandler)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

    def startTCP(self):
        if False:
            print('Hello World!')
        server = ThreadedTCPServer((self.server_address, int(self.port)), TCPHandler)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()