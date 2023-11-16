"""Representation of an EnOcean dongle."""
import glob
import logging
from os.path import basename, normpath
from enocean.communicators import SerialCommunicator
from enocean.protocol.packet import RadioPacket
import serial
from homeassistant.helpers.dispatcher import async_dispatcher_connect, dispatcher_send
from .const import SIGNAL_RECEIVE_MESSAGE, SIGNAL_SEND_MESSAGE
_LOGGER = logging.getLogger(__name__)

class EnOceanDongle:
    """Representation of an EnOcean dongle.

    The dongle is responsible for receiving the ENOcean frames,
    creating devices if needed, and dispatching messages to platforms.
    """

    def __init__(self, hass, serial_path):
        if False:
            while True:
                i = 10
        'Initialize the EnOcean dongle.'
        self._communicator = SerialCommunicator(port=serial_path, callback=self.callback)
        self.serial_path = serial_path
        self.identifier = basename(normpath(serial_path))
        self.hass = hass
        self.dispatcher_disconnect_handle = None

    async def async_setup(self):
        """Finish the setup of the bridge and supported platforms."""
        self._communicator.start()
        self.dispatcher_disconnect_handle = async_dispatcher_connect(self.hass, SIGNAL_SEND_MESSAGE, self._send_message_callback)

    def unload(self):
        if False:
            i = 10
            return i + 15
        'Disconnect callbacks established at init time.'
        if self.dispatcher_disconnect_handle:
            self.dispatcher_disconnect_handle()
            self.dispatcher_disconnect_handle = None

    def _send_message_callback(self, command):
        if False:
            return 10
        'Send a command through the EnOcean dongle.'
        self._communicator.send(command)

    def callback(self, packet):
        if False:
            i = 10
            return i + 15
        "Handle EnOcean device's callback.\n\n        This is the callback function called by python-enocan whenever there\n        is an incoming packet.\n        "
        if isinstance(packet, RadioPacket):
            _LOGGER.debug('Received radio packet: %s', packet)
            dispatcher_send(self.hass, SIGNAL_RECEIVE_MESSAGE, packet)

def detect():
    if False:
        while True:
            i = 10
    'Return a list of candidate paths for USB ENOcean dongles.\n\n    This method is currently a bit simplistic, it may need to be\n    improved to support more configurations and OS.\n    '
    globs_to_test = ['/dev/tty*FTOA2PV*', '/dev/serial/by-id/*EnOcean*']
    found_paths = []
    for current_glob in globs_to_test:
        found_paths.extend(glob.glob(current_glob))
    return found_paths

def validate_path(path: str):
    if False:
        i = 10
        return i + 15
    'Return True if the provided path points to a valid serial port, False otherwise.'
    try:
        SerialCommunicator(port=path)
        return True
    except serial.SerialException as exception:
        _LOGGER.warning('Dongle path %s is invalid: %s', path, str(exception))
        return False