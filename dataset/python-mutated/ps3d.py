import sys
import traceback
import time
import struct
import threading
import os
if os.path.exists('../../lib/python'):
    sys.path.append('../PS3BDRemote')
    sys.path.append('../../lib/python')
    from bt.hid import HID
    from bt.bt import bt_lookup_name
    from xbmcclient import XBMCClient
    from ps3 import sixaxis
    from ps3_remote import process_keys as process_remote
    try:
        from ps3 import sixwatch
    except Exception as e:
        print('Failed to import sixwatch now disabled: ' + str(e))
        sixwatch = None
    try:
        import zeroconf
    except:
        zeroconf = None
    ICON_PATH = '../../icons/'
else:
    from kodi.bt.hid import HID
    from kodi.bt.bt import bt_lookup_name
    from kodi.xbmcclient import XBMCClient
    from kodi.ps3 import sixaxis
    from kodi.ps3_remote import process_keys as process_remote
    from kodi.defs import *
    try:
        from kodi.ps3 import sixwatch
    except Exception as e:
        print('Failed to import sixwatch now disabled: ' + str(e))
        sixwatch = None
    try:
        import kodi.zeroconf as zeroconf
    except:
        zeroconf = None
event_threads = []

def printerr():
    if False:
        for i in range(10):
            print('nop')
    trace = ''
    exception = ''
    exc_list = traceback.format_exception_only(sys.exc_type, sys.exc_value)
    for entry in exc_list:
        exception += entry
    tb_list = traceback.format_tb(sys.exc_info()[2])
    for entry in tb_list:
        trace += entry
    print('%s\n%s' % (exception, trace), 'Script Error')

class StoppableThread(threading.Thread):

    def __init__(self):
        if False:
            print('Hello World!')
        threading.Thread.__init__(self)
        self._stop = False
        self.set_timeout(0)

    def stop_thread(self):
        if False:
            while True:
                i = 10
        self._stop = True

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        return self._stop

    def close_sockets(self):
        if False:
            while True:
                i = 10
        if self.isock:
            try:
                self.isock.close()
            except:
                pass
        self.isock = None
        if self.csock:
            try:
                self.csock.close()
            except:
                pass
        self.csock = None
        self.last_action = 0

    def set_timeout(self, seconds):
        if False:
            print('Hello World!')
        self.timeout = seconds

    def reset_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        self.last_action = time.time()

    def idle_time(self):
        if False:
            for i in range(10):
                print('nop')
        return time.time() - self.last_action

    def timed_out(self):
        if False:
            for i in range(10):
                print('nop')
        if time.time() - self.last_action > self.timeout:
            return True
        else:
            return False

class PS3SixaxisThread(StoppableThread):

    def __init__(self, csock, isock, ipaddr='127.0.0.1'):
        if False:
            return 10
        StoppableThread.__init__(self)
        self.csock = csock
        self.isock = isock
        self.xbmc = XBMCClient(name='PS3 Sixaxis', icon_file=ICON_PATH + '/bluetooth.png', ip=ipaddr)
        self.set_timeout(600)

    def run(self):
        if False:
            while True:
                i = 10
        six = sixaxis.sixaxis(self.xbmc, self.csock, self.isock)
        self.xbmc.connect()
        self.reset_timeout()
        try:
            while not self.stop():
                if self.timed_out():
                    raise Exception('PS3 Sixaxis powering off, timed out')
                if self.idle_time() > 50:
                    self.xbmc.connect()
                try:
                    if six.process_socket(self.isock):
                        self.reset_timeout()
                except Exception as e:
                    print(e)
                    break
        except Exception as e:
            printerr()
        six.close()
        self.close_sockets()

class PS3RemoteThread(StoppableThread):

    def __init__(self, csock, isock, ipaddr='127.0.0.1'):
        if False:
            return 10
        StoppableThread.__init__(self)
        self.csock = csock
        self.isock = isock
        self.xbmc = XBMCClient(name='PS3 Blu-Ray Remote', icon_file=ICON_PATH + '/bluetooth.png', ip=ipaddr)
        self.set_timeout(600)
        self.services = []
        self.current_xbmc = 0

    def run(self):
        if False:
            while True:
                i = 10
        self.xbmc.connect()
        try:
            try:
                self.zeroconf_thread = ZeroconfThread()
                self.zeroconf_thread.add_service('_xbmc-events._udp', self.zeroconf_service_handler)
                self.zeroconf_thread.start()
            except Exception as e:
                print(str(e))
            while not self.stop():
                status = process_remote(self.isock, self.xbmc)
                if status == 2:
                    if self.timed_out():
                        raise Exception('PS3 Blu-Ray Remote powering off, timed out')
                elif status == 3:
                    self.next_xbmc()
                elif status == 4:
                    self.previous_xbmc()
                elif not status:
                    self.reset_timeout()
        except Exception as e:
            print(str(e))
        self.zeroconf_thread.stop()
        self.close_sockets()

    def next_xbmc(self):
        if False:
            return 10
        '\n        Connect to the next XBMC instance\n        '
        self.current_xbmc = (self.current_xbmc + 1) % len(self.services)
        self.reconnect()
        return

    def previous_xbmc(self):
        if False:
            return 10
        '\n        Connect to the previous XBMC instance\n        '
        self.current_xbmc -= 1
        if self.current_xbmc < 0:
            self.current_xbmc = len(self.services) - 1
        self.reconnect()
        return

    def reconnect(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reconnect to an XBMC instance based on self.current_xbmc\n        '
        try:
            service = self.services[self.current_xbmc]
            print('Connecting to %s' % service['name'])
            self.xbmc.connect(service['address'], service['port'])
            self.xbmc.send_notification('PS3 Blu-Ray Remote', 'New Connection', None)
        except Exception as e:
            print(str(e))

    def zeroconf_service_handler(self, event, service):
        if False:
            for i in range(10):
                print('nop')
        '\n        Zeroconf event handler\n        '
        if event == zeroconf.SERVICE_FOUND:
            self.services.append(service)
        elif event == zeroconf.SERVICE_LOST:
            try:
                for s in self.services:
                    if service['name'] == s['name']:
                        self.services.remove(s)
                        break
            except:
                pass
        return

class SixWatch(threading.Thread):

    def __init__(self, mac):
        if False:
            while True:
                i = 10
        threading.Thread.__init__(self)
        self.mac = mac
        self.daemon = True
        self.start()

    def run(self):
        if False:
            print('Hello World!')
        while True:
            try:
                sixwatch.main(self.mac)
            except Exception as e:
                print('Exception caught in sixwatch, restarting: ' + str(e))

class ZeroconfThread(threading.Thread):
    """

    """

    def __init__(self):
        if False:
            while True:
                i = 10
        threading.Thread.__init__(self)
        self._zbrowser = None
        self._services = []

    def run(self):
        if False:
            i = 10
            return i + 15
        if zeroconf:
            self._zbrowser = zeroconf.Browser()
            for service in self._services:
                self._zbrowser.add_service(service[0], service[1])
            self._zbrowser.run()
        return

    def stop(self):
        if False:
            return 10
        '\n        Stop the zeroconf browser\n        '
        try:
            self._zbrowser.stop()
        except:
            pass
        return

    def add_service(self, type, handler):
        if False:
            while True:
                i = 10
        '\n        Add a new service to search for.\n        NOTE: Services must be added before thread starts.\n        '
        self._services.append([type, handler])

def usage():
    if False:
        i = 10
        return i + 15
    print('\nPS3 Sixaxis / Blu-Ray Remote HID Server v0.1\n\nUsage: ps3.py [bdaddress] [XBMC host]\n\n  bdaddress  => address of local bluetooth device to use (default: auto)\n                (e.g. aa:bb:cc:dd:ee:ff)\n  ip address => IP address or hostname of the XBMC instance (default: localhost)\n                (e.g. 192.168.1.110)\n')

def start_hidd(bdaddr=None, ipaddr='127.0.0.1'):
    if False:
        i = 10
        return i + 15
    devices = ['PLAYSTATION(R)3 Controller', 'BD Remote Control']
    hid = HID(bdaddr)
    watch = None
    if sixwatch:
        try:
            print('Starting USB sixwatch')
            watch = SixWatch(hid.get_local_address())
        except Exception as e:
            print('Failed to initialize sixwatch' + str(e))
            pass
    while True:
        if hid.listen():
            (csock, addr) = hid.get_control_socket()
            device_name = bt_lookup_name(addr[0])
            if device_name == devices[0]:
                handle_ps3_controller(hid, ipaddr)
            elif device_name == devices[1]:
                handle_ps3_remote(hid, ipaddr)
            else:
                print('Unknown Device: %s' % device_name)

def handle_ps3_controller(hid, ipaddr):
    if False:
        while True:
            i = 10
    print('Received connection from a Sixaxis PS3 Controller')
    csock = hid.get_control_socket()[0]
    isock = hid.get_interrupt_socket()[0]
    sixaxis = PS3SixaxisThread(csock, isock, ipaddr)
    add_thread(sixaxis)
    sixaxis.start()
    return

def handle_ps3_remote(hid, ipaddr):
    if False:
        while True:
            i = 10
    print('Received connection from a PS3 Blu-Ray Remote')
    csock = hid.get_control_socket()[0]
    isock = hid.get_interrupt_socket()[0]
    isock.settimeout(1)
    remote = PS3RemoteThread(csock, isock, ipaddr)
    add_thread(remote)
    remote.start()
    return

def add_thread(thread):
    if False:
        while True:
            i = 10
    global event_threads
    event_threads.append(thread)

def main():
    if False:
        return 10
    if len(sys.argv) > 3:
        return usage()
    bdaddr = ''
    ipaddr = '127.0.0.1'
    try:
        for addr in sys.argv[1:]:
            try:
                if ''.join([str(len(a)) for a in addr.split(':')]) != '222222':
                    raise Exception('Invalid format')
                bdaddr = addr
                print('Connecting to Bluetooth device: %s' % bdaddr)
            except Exception as e:
                try:
                    ipaddr = addr
                    print('Connecting to : %s' % ipaddr)
                except:
                    print(str(e))
                    return usage()
    except Exception as e:
        pass
    print('Starting HID daemon')
    start_hidd(bdaddr, ipaddr)
if __name__ == '__main__':
    try:
        main()
    finally:
        for t in event_threads:
            try:
                print('Waiting for thread ' + str(t) + ' to terminate')
                t.stop_thread()
                if t.isAlive():
                    t.join()
                print('Thread ' + str(t) + ' terminated')
            except Exception as e:
                print(str(e))
        pass