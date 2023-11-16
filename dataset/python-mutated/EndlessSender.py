import numpy as np
from urh import settings
from urh.dev.VirtualDevice import Mode, VirtualDevice
from urh.util.RingBuffer import RingBuffer

class EndlessSender(object):
    """
    Enter endless send mode for a device and send data if data gets pushed to ringbuffer.
    """

    def __init__(self, backend_handler, name: str):
        if False:
            return 10
        self.__device = VirtualDevice(backend_handler=backend_handler, name=name, mode=Mode.send)
        self.ringbuffer = RingBuffer(int(settings.CONTINUOUS_BUFFER_SIZE_MB * 10 ** 6) // 8, self.__device.data_type)
        self.__device.continuous_send_ring_buffer = self.ringbuffer
        self.__device.is_send_continuous = True

    @property
    def device(self) -> VirtualDevice:
        if False:
            print('Hello World!')
        return self.__device

    @device.setter
    def device(self, value: VirtualDevice):
        if False:
            while True:
                i = 10
        self.__device = value
        self.__device.is_send_continuous = True
        self.ringbuffer = RingBuffer(int(settings.CONTINUOUS_BUFFER_SIZE_MB * 10 ** 6) // 8, self.__device.data_type)
        self.__device.continuous_send_ring_buffer = self.ringbuffer

    @property
    def device_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.device.name

    @device_name.setter
    def device_name(self, value: str):
        if False:
            i = 10
            return i + 15
        if value != self.device_name:
            self.device = VirtualDevice(backend_handler=self.device.backend_handler, name=value, mode=Mode.send)

    def start(self):
        if False:
            return 10
        self.device.num_sending_repeats = 0
        self.device.start()

    def stop(self):
        if False:
            while True:
                i = 10
        self.device.stop('EndlessSender stopped.')

    def push_data(self, data: np.ndarray):
        if False:
            i = 10
            return i + 15
        self.ringbuffer.push(data)
if __name__ == '__main__':
    from urh.dev.BackendHandler import BackendHandler
    from urh.signalprocessing.Message import Message
    from urh.signalprocessing.MessageType import MessageType
    from urh.signalprocessing.Modulator import Modulator
    from urh.util.Logger import logger
    import time
    endless_sender = EndlessSender(BackendHandler(), 'HackRF')
    msg = Message([1, 0] * 16 + [1, 1, 0, 0] * 8 + [0, 0, 1, 1] * 8 + [1, 0, 1, 1, 1, 0, 0, 1, 1, 1] * 4, 0, MessageType('empty_message_type'))
    modulator = Modulator('test_modulator')
    modulator.samples_per_symbol = 1000
    modulator.carrier_freq_hz = 55000.0
    logger.debug('Starting endless sender')
    endless_sender.start()
    time.sleep(1)
    logger.debug('Pushing data')
    endless_sender.push_data(modulator.modulate(msg.encoded_bits))
    logger.debug('Pushed data')
    time.sleep(5)
    logger.debug('Stopping endless sender')
    endless_sender.stop()
    time.sleep(1)
    logger.debug('bye')