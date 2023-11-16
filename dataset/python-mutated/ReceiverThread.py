import numpy as np
from urh import settings
from urh.dev.gr.AbstractBaseThread import AbstractBaseThread
from urh.util.Logger import logger

class ReceiverThread(AbstractBaseThread):

    def __init__(self, frequency, sample_rate, bandwidth, gain, if_gain, baseband_gain, ip='127.0.0.1', parent=None, resume_on_full_receive_buffer=False):
        if False:
            i = 10
            return i + 15
        super().__init__(frequency, sample_rate, bandwidth, gain, if_gain, baseband_gain, True, ip, parent)
        self.resume_on_full_receive_buffer = resume_on_full_receive_buffer
        self.data = None

    def init_recv_buffer(self):
        if False:
            return 10
        n_samples = settings.get_receive_buffer_size(self.resume_on_full_receive_buffer, self.is_in_spectrum_mode)
        self.data = np.zeros(n_samples, dtype=np.complex64)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        if self.data is None:
            self.init_recv_buffer()
        self.initialize_process()
        logger.info('Initialize receive socket')
        self.init_recv_socket()
        recv = self.socket.recv
        rcvd = b''
        try:
            while not self.isInterruptionRequested():
                try:
                    rcvd += recv(32768)
                except Exception as e:
                    logger.exception(e)
                if len(rcvd) < 8:
                    self.stop('Stopped receiving: No data received anymore')
                    return
                if len(rcvd) % 8 != 0:
                    continue
                try:
                    tmp = np.fromstring(rcvd, dtype=np.complex64)
                    num_samples = len(tmp)
                    if self.data is None:
                        self.init_recv_buffer()
                    if self.current_index + num_samples >= len(self.data):
                        if self.resume_on_full_receive_buffer:
                            self.current_index = 0
                            if num_samples >= len(self.data):
                                self.stop('Receiving buffer too small.')
                        else:
                            self.stop('Receiving Buffer is full.')
                            return
                    self.data[self.current_index:self.current_index + num_samples] = tmp
                    self.current_index += num_samples
                    rcvd = b''
                except ValueError:
                    self.stop('Could not receive data. Is your Hardware ok?')
        except RuntimeError:
            logger.error('Receiver Thread crashed.')