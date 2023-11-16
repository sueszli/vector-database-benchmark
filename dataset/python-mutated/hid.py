from bluetooth import *
import fcntl
import bluetooth._bluetooth as _bt
import array

class HID:

    def __init__(self, bdaddress=None):
        if False:
            while True:
                i = 10
        self.cport = 17
        self.iport = 19
        self.backlog = 1
        self.address = ''
        if bdaddress:
            self.address = bdaddress
        self.csock = BluetoothSocket(L2CAP)
        self.csock.bind((self.address, self.cport))
        set_l2cap_mtu(self.csock, 64)
        self.csock.settimeout(2)
        self.csock.listen(self.backlog)
        self.isock = BluetoothSocket(L2CAP)
        self.isock.bind((self.address, self.iport))
        set_l2cap_mtu(self.isock, 64)
        self.isock.settimeout(2)
        self.isock.listen(self.backlog)
        self.connected = False

    def listen(self):
        if False:
            while True:
                i = 10
        try:
            (self.client_csock, self.caddress) = self.csock.accept()
            print('Accepted Control connection from %s' % self.caddress[0])
            (self.client_isock, self.iaddress) = self.isock.accept()
            print('Accepted Interrupt connection from %s' % self.iaddress[0])
            self.connected = True
            return True
        except Exception as e:
            self.connected = False
            return False

    def get_local_address(self):
        if False:
            return 10
        hci = BluetoothSocket(HCI)
        fd = hci.fileno()
        buf = array.array('B', [0] * 96)
        fcntl.ioctl(fd, _bt.HCIGETDEVINFO, buf, 1)
        data = struct.unpack_from('H8s6B', buf.tostring())
        return data[2:8][::-1]

    def get_control_socket(self):
        if False:
            while True:
                i = 10
        if self.connected:
            return (self.client_csock, self.caddress)
        else:
            return None

    def get_interrupt_socket(self):
        if False:
            return 10
        if self.connected:
            return (self.client_isock, self.iaddress)
        else:
            return None