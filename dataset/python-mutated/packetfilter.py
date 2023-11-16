from core.utils import set_ip_forwarding, iptables
from core.logger import logger
from scapy.all import *
from traceback import print_exc
from netfilterqueue import NetfilterQueue
formatter = logging.Formatter('%(asctime)s [PacketFilter] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log = logger().setup_logger('PacketFilter', formatter)

class PacketFilter:

    def __init__(self, filter):
        if False:
            while True:
                i = 10
        self.filter = filter

    def start(self):
        if False:
            i = 10
            return i + 15
        set_ip_forwarding(1)
        iptables().NFQUEUE()
        self.nfqueue = NetfilterQueue()
        self.nfqueue.bind(0, self.modify)
        self.nfqueue.run()

    def modify(self, pkt):
        if False:
            return 10
        data = pkt.get_payload()
        packet = IP(data)
        for filter in self.filter:
            try:
                execfile(filter)
            except Exception:
                log.debug('Error occurred in filter', filter)
                print_exc()
        pkt.set_payload(str(packet))
        pkt.accept()

    def stop(self):
        if False:
            return 10
        self.nfqueue.unbind()
        set_ip_forwarding(0)
        iptables().flush()