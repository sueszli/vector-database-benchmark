BLUEZ = 0
try:
    import bluetooth
    BLUEZ = 1
except:
    try:
        import lightblue
    except:
        print('ERROR: You need to have either LightBlue or PyBluez installed\n       in order to use this program.')
        print('- PyBluez (Linux / Windows XP) http://org.csail.mit.edu/pybluez/')
        print('- LightBlue (Mac OS X / Linux) http://lightblue.sourceforge.net/')
        exit()

def bt_create_socket():
    if False:
        return 10
    if BLUEZ:
        sock = bluetooth.BluetoothSocket(bluetooth.L2CAP)
    else:
        sock = lightblue.socket(lightblue.L2CAP)
    return sock

def bt_create_rfcomm_socket():
    if False:
        return 10
    if BLUEZ:
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.bind(('', bluetooth.PORT_ANY))
    else:
        sock = lightblue.socket(lightblue.RFCOMM)
        sock.bind(('', 0))
    return sock

def bt_discover_devices():
    if False:
        print('Hello World!')
    if BLUEZ:
        nearby = bluetooth.discover_devices()
    else:
        nearby = lightblue.finddevices()
    return nearby

def bt_lookup_name(bdaddr):
    if False:
        for i in range(10):
            print('nop')
    if BLUEZ:
        bname = bluetooth.lookup_name(bdaddr)
    else:
        bname = bdaddr[1]
    return bname

def bt_lookup_addr(bdaddr):
    if False:
        print('Hello World!')
    if BLUEZ:
        return bdaddr
    else:
        return bdaddr[0]

def bt_advertise(name, uuid, socket):
    if False:
        while True:
            i = 10
    if BLUEZ:
        bluetooth.advertise_service(socket, name, service_id=uuid, service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS], profiles=[bluetooth.SERIAL_PORT_PROFILE])
    else:
        lightblue.advertise(name, socket, lightblue.RFCOMM)

def bt_stop_advertising(socket):
    if False:
        for i in range(10):
            print('nop')
    if BLUEZ:
        stop_advertising(socket)
    else:
        lightblue.stopadvertise(socket)