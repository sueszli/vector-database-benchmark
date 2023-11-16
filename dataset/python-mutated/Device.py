import threading
import time
from collections import OrderedDict
from enum import Enum
from multiprocessing import Value, Process, Pipe
from multiprocessing.connection import Connection
from pickle import UnpicklingError
import numpy as np
from urh.util.Formatter import Formatter
from urh import settings
from urh.dev.native.SendConfig import SendConfig
from urh.signalprocessing.IQArray import IQArray
from urh.util import util
from urh.util.Logger import logger
util.set_shared_library_path()

class Device(object):
    JOIN_TIMEOUT = 1.0
    SYNC_TX_CHUNK_SIZE = 0
    CONTINUOUS_TX_CHUNK_SIZE = 0
    DATA_TYPE = np.float32

    class Command(Enum):
        STOP = 0
        SET_FREQUENCY = 1
        SET_SAMPLE_RATE = 2
        SET_BANDWIDTH = 3
        SET_RF_GAIN = 4
        SET_IF_GAIN = 5
        SET_BB_GAIN = 6
        SET_DIRECT_SAMPLING_MODE = 7
        SET_FREQUENCY_CORRECTION = 8
        SET_CHANNEL_INDEX = 9
        SET_ANTENNA_INDEX = 10
        SET_BIAS_TEE_ENABLED = 11
    ASYNCHRONOUS = False
    DEVICE_LIB = None
    DEVICE_METHODS = {Command.SET_FREQUENCY.name: 'set_center_freq', Command.SET_SAMPLE_RATE.name: 'set_sample_rate', Command.SET_BANDWIDTH.name: 'set_bandwidth', Command.SET_RF_GAIN.name: 'set_rf_gain', Command.SET_IF_GAIN.name: {'rx': 'set_if_rx_gain', 'tx': 'set_if_tx_gain'}, Command.SET_BB_GAIN.name: {'rx': 'set_baseband_gain'}}

    @classmethod
    def get_device_list(cls):
        if False:
            return 10
        return []

    @classmethod
    def process_command(cls, command, ctrl_connection, is_tx: bool):
        if False:
            return 10
        is_rx = not is_tx
        if command == cls.Command.STOP.name:
            return cls.Command.STOP.name
        (tag, value) = command
        try:
            if isinstance(cls.DEVICE_METHODS[tag], str):
                method_name = cls.DEVICE_METHODS[tag]
            elif isinstance(cls.DEVICE_METHODS[tag], dict):
                method_name = cls.DEVICE_METHODS[tag]['rx' if is_rx else 'tx']
            else:
                method_name = None
        except KeyError:
            method_name = None
        if method_name:
            try:
                try:
                    check_method_name = cls.DEVICE_METHODS[tag + '_get_allowed_values']
                    allowed_values = getattr(cls.DEVICE_LIB, check_method_name)()
                    next_allowed = min(allowed_values, key=lambda x: abs(x - value))
                    if value != next_allowed:
                        ctrl_connection.send('{}: {} not in range of supported values. Assuming {}'.format(tag, value, next_allowed))
                        value = next_allowed
                except (KeyError, AttributeError):
                    pass
                ret = getattr(cls.DEVICE_LIB, method_name)(value)
                if isinstance(value, int) or isinstance(value, float):
                    value = Formatter.big_value_with_suffix(value)
                ctrl_connection.send('{0} to {1}:{2}'.format(tag, value, ret))
            except AttributeError as e:
                logger.warning(str(e))

    @classmethod
    def setup_device(cls, ctrl_connection: Connection, device_identifier):
        if False:
            return 10
        raise NotImplementedError('Overwrite this method in subclass!')

    @classmethod
    def init_device(cls, ctrl_connection: Connection, is_tx: bool, parameters: OrderedDict) -> bool:
        if False:
            while True:
                i = 10
        if cls.setup_device(ctrl_connection, device_identifier=parameters['identifier']):
            for (parameter, value) in parameters.items():
                cls.process_command((parameter, value), ctrl_connection, is_tx)
            return True
        else:
            return False

    @classmethod
    def adapt_num_read_samples_to_sample_rate(cls, sample_rate: float):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Overwrite this method in subclass!')

    @classmethod
    def shutdown_device(cls, ctrl_connection, is_tx: bool):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('Overwrite this method in subclass!')

    @classmethod
    def enter_async_receive_mode(cls, data_connection: Connection, ctrl_connection: Connection) -> int:
        if False:
            print('Hello World!')
        raise NotImplementedError('Overwrite this method in subclass!')

    @classmethod
    def prepare_sync_receive(cls, ctrl_connection: Connection):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Overwrite this method in subclass!')

    @classmethod
    def receive_sync(cls, data_conn: Connection):
        if False:
            print('Hello World!')
        raise NotImplementedError('Overwrite this method in subclass!')

    @classmethod
    def enter_async_send_mode(cls, callback: object):
        if False:
            return 10
        raise NotImplementedError('Overwrite this method in subclass!')

    @classmethod
    def prepare_sync_send(cls, ctrl_connection: Connection):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Overwrite this method in subclass!')

    @classmethod
    def send_sync(cls, data):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Overwrite this method in subclass!')

    @classmethod
    def device_receive(cls, data_connection: Connection, ctrl_connection: Connection, dev_parameters: OrderedDict):
        if False:
            while True:
                i = 10
        if not cls.init_device(ctrl_connection, is_tx=False, parameters=dev_parameters):
            ctrl_connection.send('failed to start rx mode')
            return False
        try:
            cls.adapt_num_read_samples_to_sample_rate(dev_parameters[cls.Command.SET_SAMPLE_RATE.name])
        except NotImplementedError:
            pass
        if cls.ASYNCHRONOUS:
            ret = cls.enter_async_receive_mode(data_connection, ctrl_connection)
        else:
            ret = cls.prepare_sync_receive(ctrl_connection)
        if ret != 0:
            ctrl_connection.send('failed to start rx mode')
            return False
        exit_requested = False
        ctrl_connection.send('successfully started rx mode')
        while not exit_requested:
            if cls.ASYNCHRONOUS:
                try:
                    time.sleep(0.25)
                except KeyboardInterrupt:
                    pass
            else:
                cls.receive_sync(data_connection)
            while ctrl_connection.poll():
                result = cls.process_command(ctrl_connection.recv(), ctrl_connection, is_tx=False)
                if result == cls.Command.STOP.name:
                    exit_requested = True
                    break
        cls.shutdown_device(ctrl_connection, is_tx=False)
        data_connection.close()
        ctrl_connection.close()

    @classmethod
    def device_send(cls, ctrl_connection: Connection, send_config: SendConfig, dev_parameters: OrderedDict):
        if False:
            while True:
                i = 10
        if not cls.init_device(ctrl_connection, is_tx=True, parameters=dev_parameters):
            ctrl_connection.send('failed to start tx mode')
            return False
        if cls.ASYNCHRONOUS:
            ret = cls.enter_async_send_mode(send_config.get_data_to_send)
        else:
            ret = cls.prepare_sync_send(ctrl_connection)
        if ret != 0:
            ctrl_connection.send('failed to start tx mode')
            return False
        exit_requested = False
        buffer_size = cls.CONTINUOUS_TX_CHUNK_SIZE if send_config.continuous else cls.SYNC_TX_CHUNK_SIZE
        if not cls.ASYNCHRONOUS and buffer_size == 0:
            logger.warning('Send buffer size is zero!')
        ctrl_connection.send('successfully started tx mode')
        while not exit_requested and (not send_config.sending_is_finished()):
            if cls.ASYNCHRONOUS:
                try:
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    pass
            else:
                cls.send_sync(send_config.get_data_to_send(buffer_size))
            while ctrl_connection.poll():
                result = cls.process_command(ctrl_connection.recv(), ctrl_connection, is_tx=True)
                if result == cls.Command.STOP.name:
                    exit_requested = True
                    break
        if not cls.ASYNCHRONOUS:
            time.sleep(0.75)
        if exit_requested:
            logger.debug('{}: exit requested. Stopping sending'.format(cls.__class__.__name__))
        if send_config.sending_is_finished():
            logger.debug('{}: sending is finished.'.format(cls.__class__.__name__))
        cls.shutdown_device(ctrl_connection, is_tx=True)
        ctrl_connection.close()

    def __init__(self, center_freq, sample_rate, bandwidth, gain, if_gain=1, baseband_gain=1, resume_on_full_receive_buffer=False):
        if False:
            print('Hello World!')
        super().__init__()
        self.error_not_open = -4242
        self.__bandwidth = bandwidth
        self.__frequency = center_freq
        self.__gain = gain
        self.__if_gain = if_gain
        self.__baseband_gain = baseband_gain
        self.__sample_rate = sample_rate
        self.__channel_index = 0
        self.__antenna_index = 0
        self.__freq_correction = 0
        self.__bias_tee_enabled = False
        self.__direct_sampling_mode = 0
        self.bandwidth_is_adjustable = True
        self.is_in_spectrum_mode = False
        self.sending_is_continuous = False
        self.continuous_send_ring_buffer = None
        self.num_samples_to_send = None
        self._current_sent_sample = Value('L', 0)
        self._current_sending_repeat = Value('L', 0)
        self.success = 0
        self.error_codes = {}
        self.device_messages = []
        self.receive_process_function = self.device_receive
        self.send_process_function = self.device_send
        (self.parent_data_conn, self.child_data_conn) = Pipe(duplex=False)
        (self.parent_ctrl_conn, self.child_ctrl_conn) = Pipe()
        self.send_buffer = None
        self.send_buffer_reader = None
        self.device_serial = None
        self.device_number = 0
        self.samples_to_send = np.array([], dtype=self.DATA_TYPE)
        self.sending_repeats = 1
        self.resume_on_full_receive_buffer = resume_on_full_receive_buffer
        self.current_recv_index = 0
        self.is_receiving = False
        self.is_transmitting = False
        self.device_ip = '192.168.10.2'
        self.receive_buffer = None
        self.spectrum_x = None
        self.spectrum_y = None
        self.apply_dc_correction = False

    def _start_read_rcv_buffer_thread(self):
        if False:
            return 10
        self.read_recv_buffer_thread = threading.Thread(target=self.read_receiving_queue)
        self.read_recv_buffer_thread.daemon = True
        self.read_recv_buffer_thread.start()

    def _start_read_message_thread(self):
        if False:
            print('Hello World!')
        self.read_dev_msg_thread = threading.Thread(target=self.read_device_messages)
        self.read_dev_msg_thread.daemon = True
        self.read_dev_msg_thread.start()

    @property
    def has_multi_device_support(self):
        if False:
            while True:
                i = 10
        return False

    @property
    def current_sent_sample(self):
        if False:
            return 10
        return self._current_sent_sample.value // 2

    @current_sent_sample.setter
    def current_sent_sample(self, value: int):
        if False:
            while True:
                i = 10
        self._current_sent_sample.value = value * 2

    @property
    def current_sending_repeat(self):
        if False:
            i = 10
            return i + 15
        return self._current_sending_repeat.value

    @current_sending_repeat.setter
    def current_sending_repeat(self, value: int):
        if False:
            print('Hello World!')
        self._current_sending_repeat.value = value

    @property
    def device_parameters(self) -> OrderedDict:
        if False:
            return 10
        return OrderedDict([(self.Command.SET_FREQUENCY.name, self.frequency), (self.Command.SET_SAMPLE_RATE.name, self.sample_rate), (self.Command.SET_BANDWIDTH.name, self.bandwidth), (self.Command.SET_RF_GAIN.name, self.gain), (self.Command.SET_IF_GAIN.name, self.if_gain), (self.Command.SET_BB_GAIN.name, self.baseband_gain), ('identifier', self.device_serial)])

    @property
    def send_config(self) -> SendConfig:
        if False:
            while True:
                i = 10
        if self.num_samples_to_send is None:
            total_samples = len(self.send_buffer)
        else:
            total_samples = 2 * self.num_samples_to_send
        return SendConfig(self.send_buffer, self._current_sent_sample, self._current_sending_repeat, total_samples, self.sending_repeats, continuous=self.sending_is_continuous, iq_to_bytes_method=self.iq_to_bytes, continuous_send_ring_buffer=self.continuous_send_ring_buffer)

    @property
    def receive_process_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        return (self.child_data_conn, self.child_ctrl_conn, self.device_parameters)

    @property
    def send_process_arguments(self):
        if False:
            return 10
        return (self.child_ctrl_conn, self.send_config, self.device_parameters)

    def init_recv_buffer(self):
        if False:
            print('Hello World!')
        if self.receive_buffer is None:
            num_samples = settings.get_receive_buffer_size(self.resume_on_full_receive_buffer, self.is_in_spectrum_mode)
            self.receive_buffer = IQArray(None, dtype=self.DATA_TYPE, n=int(num_samples))

    def log_retcode(self, retcode: int, action: str, msg=''):
        if False:
            print('Hello World!')
        msg = str(msg)
        error_code_msg = self.error_codes[retcode] if retcode in self.error_codes else 'Error Code: ' + str(retcode)
        if retcode == self.success:
            if msg:
                formatted_message = '{0}-{1} ({2}): Success'.format(type(self).__name__, action, msg)
            else:
                formatted_message = '{0}-{1}: Success'.format(type(self).__name__, action)
            logger.info(formatted_message)
        else:
            if msg:
                formatted_message = '{0}-{1} ({4}): {2} ({3})'.format(type(self).__name__, action, error_code_msg, retcode, msg)
            else:
                formatted_message = '{0}-{1}: {2} ({3})'.format(type(self).__name__, action, error_code_msg, retcode)
            logger.error(formatted_message)
        self.device_messages.append(formatted_message)

    @property
    def received_data(self):
        if False:
            i = 10
            return i + 15
        return self.receive_buffer[:self.current_recv_index]

    @property
    def sent_data(self):
        if False:
            for i in range(10):
                print('nop')
        return self.samples_to_send[:self.current_sent_sample]

    @property
    def sending_finished(self):
        if False:
            print('Hello World!')
        return self.current_sent_sample == len(self.samples_to_send)

    @property
    def bandwidth(self):
        if False:
            i = 10
            return i + 15
        return self.__bandwidth

    @bandwidth.setter
    def bandwidth(self, value):
        if False:
            print('Hello World!')
        if not self.bandwidth_is_adjustable:
            return
        if value != self.__bandwidth:
            self.__bandwidth = value
            self.set_device_bandwidth(value)

    def set_device_bandwidth(self, bw):
        if False:
            return 10
        try:
            self.parent_ctrl_conn.send((self.Command.SET_BANDWIDTH.name, int(bw)))
        except (BrokenPipeError, OSError):
            pass

    @property
    def frequency(self):
        if False:
            while True:
                i = 10
        return self.__frequency

    @frequency.setter
    def frequency(self, value):
        if False:
            return 10
        if value != self.__frequency:
            self.__frequency = value
            self.set_device_frequency(value)

    def set_device_frequency(self, value):
        if False:
            return 10
        try:
            self.parent_ctrl_conn.send((self.Command.SET_FREQUENCY.name, int(value)))
        except (BrokenPipeError, OSError):
            pass

    @property
    def gain(self):
        if False:
            i = 10
            return i + 15
        return self.__gain

    @gain.setter
    def gain(self, value):
        if False:
            return 10
        if value != self.__gain:
            self.__gain = value
            self.set_device_gain(value)

    def set_device_gain(self, gain):
        if False:
            while True:
                i = 10
        try:
            self.parent_ctrl_conn.send((self.Command.SET_RF_GAIN.name, gain))
        except (BrokenPipeError, OSError):
            pass

    @property
    def if_gain(self):
        if False:
            return 10
        return self.__if_gain

    @if_gain.setter
    def if_gain(self, value):
        if False:
            i = 10
            return i + 15
        if value != self.__if_gain:
            self.__if_gain = value
            self.set_device_if_gain(value)

    def set_device_if_gain(self, if_gain):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.parent_ctrl_conn.send((self.Command.SET_IF_GAIN.name, if_gain))
        except (BrokenPipeError, OSError):
            pass

    @property
    def baseband_gain(self):
        if False:
            while True:
                i = 10
        return self.__baseband_gain

    @baseband_gain.setter
    def baseband_gain(self, value):
        if False:
            for i in range(10):
                print('nop')
        if value != self.__baseband_gain:
            self.__baseband_gain = value
            self.set_device_baseband_gain(value)

    def set_device_baseband_gain(self, baseband_gain):
        if False:
            while True:
                i = 10
        try:
            self.parent_ctrl_conn.send((self.Command.SET_BB_GAIN.name, baseband_gain))
        except (BrokenPipeError, OSError):
            pass

    @property
    def sample_rate(self):
        if False:
            return 10
        return self.__sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        if False:
            for i in range(10):
                print('nop')
        if value != self.__sample_rate:
            self.__sample_rate = value
            self.set_device_sample_rate(value)

    def set_device_sample_rate(self, sample_rate):
        if False:
            print('Hello World!')
        try:
            self.parent_ctrl_conn.send((self.Command.SET_SAMPLE_RATE.name, int(sample_rate)))
        except (BrokenPipeError, OSError):
            pass

    @property
    def channel_index(self) -> int:
        if False:
            i = 10
            return i + 15
        return self.__channel_index

    @channel_index.setter
    def channel_index(self, value: int):
        if False:
            for i in range(10):
                print('nop')
        if value != self.__channel_index:
            self.__channel_index = value
            self.set_device_channel_index(value)

    def set_device_channel_index(self, value):
        if False:
            print('Hello World!')
        try:
            self.parent_ctrl_conn.send((self.Command.SET_CHANNEL_INDEX.name, int(value)))
        except (BrokenPipeError, OSError):
            pass

    @property
    def antenna_index(self):
        if False:
            i = 10
            return i + 15
        return self.__antenna_index

    @antenna_index.setter
    def antenna_index(self, value):
        if False:
            while True:
                i = 10
        if value != self.__antenna_index:
            self.__antenna_index = value
            self.set_device_antenna_index(value)

    def set_device_antenna_index(self, value):
        if False:
            print('Hello World!')
        try:
            self.parent_ctrl_conn.send((self.Command.SET_ANTENNA_INDEX.name, int(value)))
        except (BrokenPipeError, OSError):
            pass

    @property
    def bias_tee_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__bias_tee_enabled

    @bias_tee_enabled.setter
    def bias_tee_enabled(self, value: bool):
        if False:
            i = 10
            return i + 15
        value = bool(value)
        if value != self.__bias_tee_enabled:
            self.__bias_tee_enabled = value
            self.set_device_bias_tee_enabled(value)

    def set_device_bias_tee_enabled(self, value):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.parent_ctrl_conn.send((self.Command.SET_BIAS_TEE_ENABLED.name, int(value)))
        except (BrokenPipeError, OSError):
            pass

    @property
    def freq_correction(self):
        if False:
            print('Hello World!')
        return self.__freq_correction

    @freq_correction.setter
    def freq_correction(self, value):
        if False:
            return 10
        if value != self.__freq_correction:
            self.__freq_correction = value
            self.set_device_freq_correction(value)

    def set_device_freq_correction(self, value):
        if False:
            i = 10
            return i + 15
        try:
            self.parent_ctrl_conn.send((self.Command.SET_FREQUENCY_CORRECTION.name, int(value)))
        except (BrokenPipeError, OSError):
            pass

    @property
    def direct_sampling_mode(self):
        if False:
            i = 10
            return i + 15
        return self.__direct_sampling_mode

    @direct_sampling_mode.setter
    def direct_sampling_mode(self, value):
        if False:
            return 10
        if value != self.__direct_sampling_mode:
            self.__direct_sampling_mode = value
            self.set_device_direct_sampling_mode(value)

    def set_device_direct_sampling_mode(self, value):
        if False:
            return 10
        try:
            self.parent_ctrl_conn.send((self.Command.SET_DIRECT_SAMPLING_MODE.name, int(value)))
        except (BrokenPipeError, OSError):
            pass

    def start_rx_mode(self):
        if False:
            return 10
        self.init_recv_buffer()
        (self.parent_data_conn, self.child_data_conn) = Pipe(duplex=False)
        (self.parent_ctrl_conn, self.child_ctrl_conn) = Pipe()
        self.is_receiving = True
        logger.info('{0}: Starting RX Mode'.format(self.__class__.__name__))
        self.receive_process = Process(target=self.receive_process_function, args=self.receive_process_arguments)
        self.receive_process.daemon = True
        self._start_read_rcv_buffer_thread()
        self._start_read_message_thread()
        try:
            self.receive_process.start()
        except OSError as e:
            logger.error(repr(e))
            self.device_messages.append(repr(e))

    def stop_rx_mode(self, msg):
        if False:
            print('Hello World!')
        try:
            self.parent_ctrl_conn.send(self.Command.STOP.name)
        except (BrokenPipeError, OSError) as e:
            logger.debug('Closing parent control connection: ' + str(e))
        logger.info('{0}: Stopping RX Mode: {1}'.format(self.__class__.__name__, msg))
        if hasattr(self, 'receive_process') and self.receive_process.is_alive():
            self.receive_process.join(self.JOIN_TIMEOUT)
            if self.receive_process.is_alive():
                logger.warning('{0}: Receive process is still alive, terminating it'.format(self.__class__.__name__))
                self.receive_process.terminate()
                self.receive_process.join()
        self.is_receiving = False
        for connection in (self.parent_ctrl_conn, self.parent_data_conn, self.child_ctrl_conn, self.child_data_conn):
            try:
                connection.close()
            except OSError as e:
                logger.exception(e)

    def start_tx_mode(self, samples_to_send: np.ndarray=None, repeats=None, resume=False):
        if False:
            i = 10
            return i + 15
        self.is_transmitting = True
        (self.parent_ctrl_conn, self.child_ctrl_conn) = Pipe()
        self.init_send_parameters(samples_to_send, repeats, resume=resume)
        logger.info('{0}: Starting TX Mode'.format(self.__class__.__name__))
        self.transmit_process = Process(target=self.send_process_function, args=self.send_process_arguments)
        self.transmit_process.daemon = True
        self._start_read_message_thread()
        self.transmit_process.start()

    def stop_tx_mode(self, msg):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.parent_ctrl_conn.send(self.Command.STOP.name)
        except (BrokenPipeError, OSError) as e:
            logger.debug('Closing parent control connection: ' + str(e))
        logger.info('{0}: Stopping TX Mode: {1}'.format(self.__class__.__name__, msg))
        if hasattr(self, 'transmit_process') and self.transmit_process.is_alive():
            self.transmit_process.join(self.JOIN_TIMEOUT)
            if self.transmit_process.is_alive():
                logger.warning('{0}: Transmit process is still alive, terminating it'.format(self.__class__.__name__))
                self.transmit_process.terminate()
                self.transmit_process.join()
        self.is_transmitting = False
        try:
            self.parent_ctrl_conn.close()
        except OSError as e:
            logger.exception(e)
        try:
            self.child_ctrl_conn.close()
        except OSError as e:
            logger.exception(e)

    @staticmethod
    def bytes_to_iq(buffer) -> np.ndarray:
        if False:
            print('Hello World!')
        pass

    @staticmethod
    def iq_to_bytes(complex_samples: np.ndarray):
        if False:
            i = 10
            return i + 15
        pass

    def read_device_messages(self):
        if False:
            i = 10
            return i + 15
        while self.is_receiving or self.is_transmitting:
            try:
                message = self.parent_ctrl_conn.recv()
                try:
                    splitted = message.split(':')
                    action = ':'.join(splitted[:-1])
                    return_code = splitted[-1]
                    self.log_retcode(int(return_code), action)
                except ValueError:
                    self.device_messages.append('{0}: {1}'.format(self.__class__.__name__, message))
            except (EOFError, UnpicklingError, OSError, ConnectionResetError) as e:
                logger.info('Exiting read device message thread due to ' + str(e))
                break
        self.is_transmitting = False
        self.is_receiving = False
        logger.debug('Exiting read device errors thread')

    def read_receiving_queue(self):
        if False:
            return 10
        while self.is_receiving:
            try:
                byte_buffer = self.parent_data_conn.recv_bytes()
                samples = self.bytes_to_iq(byte_buffer)
                n_samples = len(samples)
                if n_samples == 0:
                    continue
                if self.apply_dc_correction:
                    samples = samples - np.mean(samples, axis=0)
            except OSError as e:
                logger.exception(e)
                continue
            except EOFError:
                logger.info('EOF Error: Ending receive thread')
                break
            if self.current_recv_index + n_samples >= len(self.receive_buffer):
                if self.resume_on_full_receive_buffer:
                    self.current_recv_index = 0
                    if n_samples >= len(self.receive_buffer):
                        n_samples = len(self.receive_buffer) - 1
                else:
                    self.stop_rx_mode('Receiving buffer is full {0}/{1}'.format(self.current_recv_index + n_samples, len(self.receive_buffer)))
                    return
            self.receive_buffer[self.current_recv_index:self.current_recv_index + n_samples] = samples[:n_samples]
            self.current_recv_index += n_samples
        logger.debug('Exiting read_receive_queue thread.')

    def init_send_parameters(self, samples_to_send: IQArray=None, repeats: int=None, resume=False):
        if False:
            i = 10
            return i + 15
        if samples_to_send is not None:
            if isinstance(samples_to_send, IQArray):
                samples_to_send = samples_to_send.convert_to(self.DATA_TYPE)
            else:
                samples_to_send = IQArray(samples_to_send).convert_to(self.DATA_TYPE)
            self.samples_to_send = samples_to_send
            self.send_buffer = None
        if self.send_buffer is None:
            if isinstance(self.samples_to_send, IQArray):
                self.send_buffer = self.iq_to_bytes(self.samples_to_send.data)
            else:
                self.send_buffer = self.iq_to_bytes(self.samples_to_send)
        elif not resume:
            self.current_sending_repeat = 0
        if repeats is not None:
            self.sending_repeats = repeats