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
    packet = PacketHELO('Email Notifier', ICON_NONE)
    packet.send(sock, addr)
    time.sleep(5)
    packet = PacketNOTIFICATION('New Mail!', 'RE: Check this out', ICON_PNG, ICON_PATH + '/mail.png')
    packet.send(sock, addr)
    packet = PacketBYE()
    packet.send(sock, addr)
if __name__ == '__main__':
    main()