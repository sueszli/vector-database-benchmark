import os
from threading import Thread, Event
from pocsuite3.lib.core.common import data_to_stdout, mosaic
from pocsuite3.lib.core.data import logger
import logging
os.environ['MPLBACKEND'] = 'Agg'
logging.getLogger('scapy').setLevel(logging.ERROR)
from scapy.all import WINDOWS, get_if_list, get_if_addr, AsyncSniffer

class Sniffer(Thread):

    def __init__(self, filter):
        if False:
            print('Hello World!')
        super().__init__()
        self.filter = 'host %s' % filter
        self.daemon = True
        self.socket = None
        self.use_pcap = True
        self.is_admin = False
        logger.info('Local network adapter information, choose a network you want to capture.')
        message = '----- Local IP Address -----\n'
        ifaces = []
        if WINDOWS:
            import ctypes
            from scapy.all import IFACES
            if ctypes.windll.shell32.IsUserAnAdmin():
                self.is_admin = True
            for (i, iface) in enumerate(sorted(IFACES)):
                dev = IFACES[iface]
                ifaces.append(dev.description)
                message += '{0}   {1}    {2}\n'.format(i, dev.description, mosaic(dev.ip))
        else:
            if os.getuid() == 0:
                self.is_admin = True
            ifaces = get_if_list()
            for (i, iface) in enumerate(ifaces):
                ip = get_if_addr(iface)
                message += '{0}   {1}    {2}\n'.format(i, iface, mosaic(ip))
        data_to_stdout(message)
        choose = input('Choose>: ').strip()
        self.interface = ifaces[int(choose)]
        self.use_pcap = True
        self.stop_sniffer = Event()
        self.pcap = None

    def run(self):
        if False:
            i = 10
            return i + 15
        self.pcap = AsyncSniffer()
        self.pcap._run(iface=self.interface, filter=self.filter)

    def join(self, timeout=None):
        if False:
            i = 10
            return i + 15
        self.pcap.continue_sniff = False
        super().join(timeout)