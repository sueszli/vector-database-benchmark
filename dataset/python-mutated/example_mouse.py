import os
from socket import *
import sys
if os.path.exists('../../lib/python'):
    sys.path.append('../../lib/python')
    from xbmcclient import *
    ICON_PATH = '../../icons/'
else:
    from kodi.xbmcclient import *
    from kodi.defs import *

def main():
    if False:
        i = 10
        return i + 15
    import time
    import sys
    host = 'localhost'
    port = 9777
    addr = (host, port)
    sock = socket(AF_INET, SOCK_DGRAM)
    packet = PacketHELO('Example Mouse', ICON_PNG, ICON_PATH + '/mouse.png')
    packet.send(sock, addr)
    time.sleep(2)
    for i in range(0, 65535, 2):
        packet = PacketMOUSE(i, i)
        packet.send(sock, addr)
    packet = PacketBYE()
    packet.send(sock, addr)
if __name__ == '__main__':
    main()