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
        while True:
            i = 10
    import time
    import sys
    host = 'localhost'
    port = 9777
    addr = (host, port)
    sock = socket(AF_INET, SOCK_DGRAM)
    packet = PacketHELO(devicename='Example Remote', icon_type=ICON_PNG, icon_file=ICON_PATH + '/bluetooth.png')
    packet.send(sock, addr)
    time.sleep(5)
    packet = PacketBUTTON(code='S', queue=1)
    packet.send(sock, addr)
    time.sleep(2)
    packet = PacketBUTTON(code=13, queue=1)
    packet.send(sock, addr)
    packet = PacketBYE()
    packet.send(sock, addr)
if __name__ == '__main__':
    main()