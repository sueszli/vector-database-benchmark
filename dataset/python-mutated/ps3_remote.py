import sys
try:
    sys.path.append('../../lib/python')
    from xbmcclient import *
    from ps3.keymaps import keymap_remote as g_keymap
    from bt.bt import *
    ICON_PATH = '../../icons/'
except:
    from kodi.xbmcclient import *
    from kodi.ps3.keymaps import keymap_remote as g_keymap
    from kodi.bt.bt import *
    from kodi.defs import *
import os
import time
xbmc = None
bticon = ICON_PATH + '/bluetooth.png'

def get_remote_address(remote, target_name='BD Remote Control'):
    if False:
        while True:
            i = 10
    global xbmc
    target_connected = False
    target_address = None
    while target_connected is False:
        xbmc.send_notification('Action Required!', 'Hold Start+Enter on your remote.', bticon)
        print('Searching for %s' % target_name)
        print('(Hold Start + Enter on remote to make it discoverable)')
        time.sleep(2)
        if not target_address:
            try:
                nearby_devices = bt_discover_devices()
            except Exception as e:
                print('Error performing bluetooth discovery')
                print(str(e))
                xbmc.send_notification('Error', 'Unable to find devices.', bticon)
                time.sleep(5)
                continue
            for bdaddr in nearby_devices:
                bname = bt_lookup_name(bdaddr)
                addr = bt_lookup_addr(bdaddr)
                print('%s (%s) in range' % (bname, addr))
                if target_name == bname:
                    target_address = addr
                    break
        if target_address is not None:
            print('Found %s with address %s' % (target_name, target_address))
            xbmc.send_notification('Found Device', 'Pairing %s, please wait.' % target_name, bticon)
            print('Attempting to pair with remote')
            try:
                remote.connect((target_address, 19))
                target_connected = True
                print('Remote Paired.\x07')
                xbmc.send_notification('Pairing Successful', 'Your remote was successfully paired and is ready to be used.', bticon)
            except:
                del remote
                remote = bt_create_socket()
                target_address = None
                xbmc.send_notification('Pairing Failed', 'An error occurred while attempting to pair.', bticon)
                print('ERROR - Could Not Connect. Trying again...')
                time.sleep(2)
        else:
            xbmc.send_notification('Error', 'No remotes were found.', bticon)
            print('Could not find BD Remote Control. Trying again...')
            time.sleep(2)
    return (remote, target_address)

def usage():
    if False:
        return 10
    print('\nPS3 Blu-Ray Remote Control Client for XBMC v0.1\n\nUsage: ps3_remote.py <address> [port]\n\n  address => address of system that XBMC is running on\n             ("localhost" if it is this machine)\n\n     port => port to send packets to\n             (default 9777)\n')

def process_keys(remote, xbmc):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return codes:\n    0 - key was processed normally\n    2 - socket read timeout\n    3 - PS and then Skip Plus was pressed (sequentially)\n    4 - PS and then Skip Minus was pressed (sequentially)\n\n    FIXME: move to enums\n    '
    done = 0
    try:
        xbmc.previous_key
    except:
        xbmc.previous_key = ''
    xbmc.connect()
    datalen = 0
    try:
        data = remote.recv(1024)
        datalen = len(data)
    except Exception as e:
        if str(e) == 'timed out':
            return 2
        time.sleep(2)
        raise e
    if datalen == 13:
        keycode = data.hex()[10:12]
        if keycode == 'ff':
            xbmc.release_button()
            return done
        try:
            if xbmc.previous_key == '43':
                xbmc.previous_key = keycode
                if keycode == '31':
                    return 3
                elif keycode == '30':
                    return 4
            xbmc.previous_key = keycode
            if g_keymap[keycode]:
                xbmc.send_remote_button(g_keymap[keycode])
        except Exception as e:
            print('Unknown data: %s' % str(e))
    return done

def main():
    if False:
        i = 10
        return i + 15
    global xbmc, bticon
    host = '127.0.0.1'
    port = 9777
    if len(sys.argv) > 1:
        try:
            host = sys.argv[1]
            port = sys.argv[2]
        except:
            pass
    else:
        return usage()
    loop_forever = True
    xbmc = XBMCClient('PS3 Bluetooth Remote', icon_file=bticon)
    while loop_forever is True:
        target_connected = False
        remote = bt_create_socket()
        xbmc.connect(host, port)
        (remote, target_address) = get_remote_address(remote)
        while True:
            if process_keys(remote, xbmc):
                break
        print('Disconnected.')
        try:
            remote.close()
        except:
            print('Cannot close.')
if __name__ == '__main__':
    main()