from impacket import structure
O_ETH = 0
O_IP = 1
O_ARP = 1
O_UDP = 2
O_TCP = 2
O_ICMP = 2
O_UDP_DATA = 3
O_ICMP_DATA = 3
MAGIC = '"ÔÃ²¡'

class PCapFileHeader(structure.Structure):
    structure = (('magic', MAGIC), ('versionMajor', '<H=2'), ('versionMinor', '<H=4'), ('GMT2localCorrection', '<l=0'), ('timeAccuracy', '<L=0'), ('maxLength', '<L=0xffff'), ('linkType', '<L=1'), ('packets', '*:=[]'))

class PCapFilePacket(structure.Structure):
    structure = (('tsec', '<L=0'), ('tmsec', '<L=0'), ('savedLength', '<L-data'), ('realLength', '<L-data'), ('data', ':'))

    def __init__(self, *args, **kargs):
        if False:
            print('Hello World!')
        structure.Structure.__init__(self, *args, **kargs)
        self['data'] = b''

class PcapFile:

    def __init__(self, fileName=None, mode='rb'):
        if False:
            return 10
        if fileName is not None:
            self.file = open(fileName, mode)
        self.hdr = None
        self.wroteHeader = False

    def reset(self):
        if False:
            return 10
        self.hdr = None
        self.file.seek(0)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.file.close()

    def fileno(self):
        if False:
            i = 10
            return i + 15
        return self.file.fileno()

    def setFile(self, file):
        if False:
            i = 10
            return i + 15
        self.file = file

    def setSnapLen(self, snapLen):
        if False:
            i = 10
            return i + 15
        self.createHeaderOnce()
        self.hdr['maxLength'] = snapLen

    def getSnapLen(self):
        if False:
            while True:
                i = 10
        self.readHeaderOnce()
        return self.hdr['maxLength']

    def setLinkType(self, linkType):
        if False:
            i = 10
            return i + 15
        self.createHeaderOnce()
        self.hdr['linkType'] = linkType

    def getLinkType(self):
        if False:
            i = 10
            return i + 15
        self.readHeaderOnce()
        return self.hdr['linkType']

    def readHeaderOnce(self):
        if False:
            for i in range(10):
                print('nop')
        if self.hdr is None:
            self.hdr = PCapFileHeader.fromFile(self.file)

    def createHeaderOnce(self):
        if False:
            print('Hello World!')
        if self.hdr is None:
            self.hdr = PCapFileHeader()

    def writeHeaderOnce(self):
        if False:
            while True:
                i = 10
        if not self.wroteHeader:
            self.wroteHeader = True
            self.file.seek(0)
            self.createHeaderOnce()
            self.file.write(self.hdr.getData())

    def read(self):
        if False:
            return 10
        self.readHeaderOnce()
        try:
            pkt = PCapFilePacket.fromFile(self.file)
            pkt['data'] = self.file.read(pkt['savedLength'])
            return pkt
        except:
            return None

    def write(self, pkt):
        if False:
            while True:
                i = 10
        self.writeHeaderOnce()
        self.file.write(str(pkt))

    def packets(self):
        if False:
            while True:
                i = 10
        self.reset()
        while 1:
            answer = self.read()
            if answer is None:
                break
            yield answer