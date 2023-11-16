from scapy.packet import Packet

class OBD_Packet(Packet):

    def extract_padding(self, s):
        if False:
            print('Hello World!')
        return ('', s)