import sys
from threading import Thread
import pcapy
from pcapy import findalldevs, open_live
from impacket.ImpactDecoder import EthDecoder, LinuxSLLDecoder

class DecoderThread(Thread):

    def __init__(self, pcapObj):
        if False:
            for i in range(10):
                print('nop')
        datalink = pcapObj.datalink()
        if pcapy.DLT_EN10MB == datalink:
            self.decoder = EthDecoder()
        elif pcapy.DLT_LINUX_SLL == datalink:
            self.decoder = LinuxSLLDecoder()
        else:
            raise Exception('Datalink type not supported: ' % datalink)
        self.pcap = pcapObj
        Thread.__init__(self)

    def run(self):
        if False:
            i = 10
            return i + 15
        self.pcap.loop(0, self.packetHandler)

    def packetHandler(self, hdr, data):
        if False:
            return 10
        print(self.decoder.decode(data))

def getInterface():
    if False:
        print('Hello World!')
    ifs = findalldevs()
    if 0 == len(ifs):
        print("You don't have enough permissions to open any interface on this system.")
        sys.exit(1)
    elif 1 == len(ifs):
        print('Only one interface present, defaulting to it.')
        return ifs[0]
    count = 0
    for iface in ifs:
        print('%i - %s' % (count, iface))
        count += 1
    idx = int(input('Please select an interface: '))
    return ifs[idx]

def main(filter):
    if False:
        for i in range(10):
            print('nop')
    dev = getInterface()
    p = open_live(dev, 1500, 0, 100)
    p.setfilter(filter)
    print('Listening on %s: net=%s, mask=%s, linktype=%d' % (dev, p.getnet(), p.getmask(), p.datalink()))
    DecoderThread(p).start()
filter = ''
if len(sys.argv) > 1:
    filter = ' '.join(sys.argv[1:])
main(filter)