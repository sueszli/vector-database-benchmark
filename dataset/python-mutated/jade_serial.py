import serial
import logging
logger = logging.getLogger('jade.serial')

class JadeSerialImpl:

    def __init__(self, device, baud, timeout):
        if False:
            print('Hello World!')
        self.device = device
        self.baud = baud
        self.timeout = timeout
        self.ser = None

    def connect(self):
        if False:
            print('Hello World!')
        assert self.ser is None
        logger.info('Connecting to {} at {}'.format(self.device, self.baud))
        self.ser = serial.Serial(self.device, self.baud, timeout=self.timeout, write_timeout=self.timeout)
        assert self.ser is not None
        if not self.ser.is_open:
            self.ser.open()
        self.ser.setRTS(False)
        self.ser.setDTR(False)
        logger.info('Connected')

    def disconnect(self):
        if False:
            i = 10
            return i + 15
        assert self.ser is not None
        self.ser.setRTS(False)
        self.ser.setDTR(False)
        self.ser.close()
        self.ser = None

    def write(self, bytes_):
        if False:
            while True:
                i = 10
        assert self.ser is not None
        return self.ser.write(bytes_)

    def read(self, n):
        if False:
            for i in range(10):
                print('nop')
        assert self.ser is not None
        return self.ser.read(n)