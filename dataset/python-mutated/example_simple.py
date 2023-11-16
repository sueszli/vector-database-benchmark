import os
from socket import *
import sys
import time
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
    host = 'localhost'
    port = 9777
    xbmc = XBMCClient('Example Remote', ICON_PATH + '/bluetooth.png')
    xbmc.connect()
    time.sleep(5)
    xbmc.send_button(map='XG', button='dpadup')
    time.sleep(5)
    xbmc.send_keyboard_button('right')
    time.sleep(5)
    xbmc.release_button()
    xbmc.close()
if __name__ == '__main__':
    main()