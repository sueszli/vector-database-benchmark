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
    host = 'localhost'
    port = 9777
    xbmc = XBMCClient('Example Remote', ICON_PATH + '/bluetooth.png')
    xbmc.connect()
    try:
        xbmc.send_action(sys.argv[2], ACTION_BUTTON)
    except:
        try:
            xbmc.send_action(sys.argv[1], ACTION_EXECBUILTIN)
        except Exception as e:
            print(str(e))
            xbmc.send_action('ActivateWindow(ShutdownMenu)')
    xbmc.close()
if __name__ == '__main__':
    main()