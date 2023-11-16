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
        for i in range(10):
            print('nop')
    import time
    import sys
    host = 'localhost'
    port = 9777
    addr = (host, port)
    sock = socket(AF_INET, SOCK_DGRAM)
    packet = PacketHELO('Example Remote', ICON_PNG, ICON_PATH + '/bluetooth.png')
    packet.send(sock, addr)
    time.sleep(5)
    packet = PacketBUTTON(map_name='XG', button_name='dpadup')
    packet.send(sock, addr)
    time.sleep(5)
    packet = PacketBUTTON(code=40)
    packet.send(sock, addr)
    time.sleep(5)
    packet = PacketBUTTON(map_name='KB', button_name='right')
    packet.send(sock, addr)
    time.sleep(5)
    packet = PacketBUTTON(code=40, down=0)
    packet.send(sock, addr)
    packet = PacketBYE()
    packet.send(sock, addr)
if __name__ == '__main__':
    main()