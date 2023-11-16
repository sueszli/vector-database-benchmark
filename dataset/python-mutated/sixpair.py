import sys
import usb
vendor = 1356
product = 616
timeout = 5000
passed_value = 1013

def find_sixaxes():
    if False:
        return 10
    res = []
    for bus in usb.busses():
        for dev in bus.devices:
            if dev.idVendor == vendor and dev.idProduct == product:
                res.append(dev)
    return res

def find_interface(dev):
    if False:
        while True:
            i = 10
    for cfg in dev.configurations:
        for itf in cfg.interfaces:
            for alt in itf:
                if alt.interfaceClass == 3:
                    return alt
    raise Exception('Unable to find interface')

def mac_to_string(mac):
    if False:
        while True:
            i = 10
    return '%02x:%02x:%02x:%02x:%02x:%02x' % (mac[0], mac[1], mac[2], mac[3], mac[4], mac[5])

def set_pair_filename(dirname, filename, mac):
    if False:
        i = 10
        return i + 15
    for bus in usb.busses():
        if int(bus.dirname) == int(dirname):
            for dev in bus.devices:
                if int(dev.filename) == int(filename):
                    if dev.idVendor == vendor and dev.idProduct == product:
                        update_pair(dev, mac)
                        return
                    else:
                        raise Exception('Device is not a sixaxis')
    raise Exception('Device not found')

def set_pair(dev, mac):
    if False:
        i = 10
        return i + 15
    itf = find_interface(dev)
    handle = dev.open()
    msg = (1, 0) + mac
    try:
        handle.detachKernelDriver(itf.interfaceNumber)
    except usb.USBError:
        pass
    handle.claimInterface(itf.interfaceNumber)
    try:
        handle.controlMsg(usb.ENDPOINT_OUT | usb.TYPE_CLASS | usb.RECIP_INTERFACE, usb.REQ_SET_CONFIGURATION, msg, passed_value, itf.interfaceNumber, timeout)
    finally:
        handle.releaseInterface()

def get_pair(dev):
    if False:
        print('Hello World!')
    itf = find_interface(dev)
    handle = dev.open()
    try:
        handle.detachKernelDriver(itf.interfaceNumber)
    except usb.USBError:
        pass
    handle.claimInterface(itf.interfaceNumber)
    try:
        msg = handle.controlMsg(usb.ENDPOINT_IN | usb.TYPE_CLASS | usb.RECIP_INTERFACE, usb.REQ_CLEAR_FEATURE, 8, passed_value, itf.interfaceNumber, timeout)
    finally:
        handle.releaseInterface()
    return msg[2:8]

def set_pair_all(mac):
    if False:
        while True:
            i = 10
    devs = find_sixaxes()
    for dev in devs:
        update_pair(dev, mac)

def update_pair(dev, mac):
    if False:
        for i in range(10):
            print('nop')
    old = get_pair(dev)
    if old != mac:
        print('Re-pairing sixaxis from:' + mac_to_string(old) + ' to:' + mac_to_string(mac))
    set_pair(dev, mac)
if __name__ == '__main__':
    devs = find_sixaxes()
    mac = None
    if len(sys.argv) > 1:
        try:
            mac = sys.argv[1].split(':')
            mac = tuple([int(x, 16) for x in mac])
            if len(mac) != 6:
                print('Invalid length of HCI address, should be 6 parts')
                mac = None
        except:
            print('Failed to parse HCI address')
            mac = None
    for dev in devs:
        if mac:
            update_pair(dev, mac)
        else:
            print('Found sixaxis paired to: ' + mac_to_string(get_pair(dev)))