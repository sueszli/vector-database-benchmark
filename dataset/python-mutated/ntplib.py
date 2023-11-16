"""Python NTP library.

Implementation of client-side NTP (RFC-1305), and useful NTP-related
functions.
"""
__all__ = ('NTPException', 'NTP', 'NTPPacket', 'NTPClient', 'NTPStats')
import datetime
import socket
import struct
import time

class NTPException(Exception):
    """Exception raised by this module."""
    __slots__ = ()

class NTP(object):
    """Helper class defining constants."""
    _SYSTEM_EPOCH = datetime.date(*time.gmtime(0)[0:3])
    'system epoch'
    _NTP_EPOCH = datetime.date(1900, 1, 1)
    'NTP epoch'
    NTP_DELTA = (_SYSTEM_EPOCH - _NTP_EPOCH).days * 24 * 3600
    'delta between system and NTP time'
    REF_ID_TABLE = {'GOES': 'Geostationary Orbit Environment Satellite', 'GPS\x00': 'Global Position System', 'GAL\x00': 'Galileo Positioning System', 'PPS\x00': 'Generic pulse-per-second', 'IRIG': 'Inter-Range Instrumentation Group', 'WWVB': 'LF Radio WWVB Ft. Collins, CO 60 kHz', 'DCF\x00': 'LF Radio DCF77 Mainflingen, DE 77.5 kHz', 'HBG\x00': 'LF Radio HBG Prangins, HB 75 kHz', 'MSF\x00': 'LF Radio MSF Anthorn, UK 60 kHz', 'JJY\x00': 'LF Radio JJY Fukushima, JP 40 kHz, Saga, JP 60 kHz', 'LORC': 'MF Radio LORAN C station, 100 kHz', 'TDF\x00': 'MF Radio Allouis, FR 162 kHz', 'CHU\x00': 'HF Radio CHU Ottawa, Ontario', 'WWV\x00': 'HF Radio WWV Ft. Collins, CO', 'WWVH': 'HF Radio WWVH Kauai, HI', 'NIST': 'NIST telephone modem', 'ACTS': 'NIST telephone modem', 'USNO': 'USNO telephone modem', 'PTB\x00': 'European telephone modem', 'LOCL': 'uncalibrated local clock', 'CESM': 'calibrated Cesium clock', 'RBDM': 'calibrated Rubidium clock', 'OMEG': 'OMEGA radionavigation system', 'DCN\x00': 'DCN routing protocol', 'TSP\x00': 'TSP time protocol', 'DTS\x00': 'Digital Time Service', 'ATOM': 'Atomic clock (calibrated)', 'VLF\x00': 'VLF radio (OMEGA,, etc.)', '1PPS': 'External 1 PPS input', 'FREE': '(Internal clock)', 'INIT': '(Initialization)', '\x00\x00\x00\x00': 'NULL'}
    'reference identifier table'
    STRATUM_TABLE = {0: 'unspecified or invalid', 1: 'primary reference (%s)'}
    'stratum table'
    MODE_TABLE = {0: 'reserved', 1: 'symmetric active', 2: 'symmetric passive', 3: 'client', 4: 'server', 5: 'broadcast', 6: 'reserved for NTP control messages', 7: 'reserved for private use'}
    'mode table'
    LEAP_TABLE = {0: 'no warning', 1: 'last minute of the day has 61 seconds', 2: 'last minute of the day has 59 seconds', 3: 'unknown (clock unsynchronized)'}
    'leap indicator table'
    __slots__ = ()

class NTPPacket(object):
    """NTP packet class.

    This represents an NTP packet.
    """
    _PACKET_FORMAT = '!B B B b 11I'
    'packet format to pack/unpack'
    __slots__ = ('leap', 'version', 'mode', 'stratum', 'poll', 'precision', 'root_delay', 'root_dispersion', 'ref_id', 'ref_timestamp', 'orig_timestamp', 'recv_timestamp', 'tx_timestamp')

    def __init__(self, version=2, mode=3, tx_timestamp=0):
        if False:
            while True:
                i = 10
        'Constructor.\n\n        Parameters:\n        version      -- NTP version\n        mode         -- packet mode (client, server)\n        tx_timestamp -- packet transmit timestamp\n        '
        self.leap = 0
        'leap second indicator'
        self.version = version
        'version'
        self.mode = mode
        'mode'
        self.stratum = 0
        'stratum'
        self.poll = 0
        'poll interval'
        self.precision = 0
        'precision'
        self.root_delay = 0
        'root delay'
        self.root_dispersion = 0
        'root dispersion'
        self.ref_id = 0
        'reference clock identifier'
        self.ref_timestamp = 0
        'reference timestamp'
        self.orig_timestamp = 0
        'originate timestamp'
        self.recv_timestamp = 0
        'receive timestamp'
        self.tx_timestamp = tx_timestamp
        'tansmit timestamp'

    def to_data(self):
        if False:
            for i in range(10):
                print('nop')
        'Convert this NTPPacket to a buffer that can be sent over a socket.\n\n        Returns:\n        buffer representing this packet\n\n        Raises:\n        NTPException -- in case of invalid field\n        '
        try:
            packed = struct.pack(NTPPacket._PACKET_FORMAT, self.leap << 6 | self.version << 3 | self.mode, self.stratum, self.poll, self.precision, _to_int(self.root_delay) << 16 | _to_frac(self.root_delay, 16), _to_int(self.root_dispersion) << 16 | _to_frac(self.root_dispersion, 16), self.ref_id, _to_int(self.ref_timestamp), _to_frac(self.ref_timestamp), _to_int(self.orig_timestamp), _to_frac(self.orig_timestamp), _to_int(self.recv_timestamp), _to_frac(self.recv_timestamp), _to_int(self.tx_timestamp), _to_frac(self.tx_timestamp))
        except struct.error:
            raise NTPException('Invalid NTP packet fields.')
        return packed

    def from_data(self, data):
        if False:
            i = 10
            return i + 15
        'Populate this instance from a NTP packet payload received from\n        the network.\n\n        Parameters:\n        data -- buffer payload\n\n        Raises:\n        NTPException -- in case of invalid packet format\n        '
        try:
            unpacked = struct.unpack(NTPPacket._PACKET_FORMAT, data[0:struct.calcsize(NTPPacket._PACKET_FORMAT)])
        except struct.error:
            raise NTPException('Invalid NTP packet.')
        self.leap = unpacked[0] >> 6 & 3
        self.version = unpacked[0] >> 3 & 7
        self.mode = unpacked[0] & 7
        self.stratum = unpacked[1]
        self.poll = unpacked[2]
        self.precision = unpacked[3]
        self.root_delay = float(unpacked[4]) / 2 ** 16
        self.root_dispersion = float(unpacked[5]) / 2 ** 16
        self.ref_id = unpacked[6]
        self.ref_timestamp = _to_time(unpacked[7], unpacked[8])
        self.orig_timestamp = _to_time(unpacked[9], unpacked[10])
        self.recv_timestamp = _to_time(unpacked[11], unpacked[12])
        self.tx_timestamp = _to_time(unpacked[13], unpacked[14])

class NTPStats(NTPPacket):
    """NTP statistics.

    Wrapper for NTPPacket, offering additional statistics like offset and
    delay, and timestamps converted to system time.
    """
    __slots__ = ['dest_timestamp']

    def __init__(self):
        if False:
            while True:
                i = 10
        'Constructor.'
        NTPPacket.__init__(self)
        self.dest_timestamp = 0
        'destination timestamp'

    @property
    def offset(self):
        if False:
            while True:
                i = 10
        'offset'
        return (self.recv_timestamp - self.orig_timestamp + (self.tx_timestamp - self.dest_timestamp)) / 2

    @property
    def delay(self):
        if False:
            while True:
                i = 10
        'round-trip delay'
        return self.dest_timestamp - self.orig_timestamp - (self.tx_timestamp - self.recv_timestamp)

    @property
    def tx_time(self):
        if False:
            return 10
        'Transmit timestamp in system time.'
        return ntp_to_system_time(self.tx_timestamp)

    @property
    def recv_time(self):
        if False:
            while True:
                i = 10
        'Receive timestamp in system time.'
        return ntp_to_system_time(self.recv_timestamp)

    @property
    def orig_time(self):
        if False:
            i = 10
            return i + 15
        'Originate timestamp in system time.'
        return ntp_to_system_time(self.orig_timestamp)

    @property
    def ref_time(self):
        if False:
            for i in range(10):
                print('nop')
        'Reference timestamp in system time.'
        return ntp_to_system_time(self.ref_timestamp)

    @property
    def dest_time(self):
        if False:
            print('Hello World!')
        'Destination timestamp in system time.'
        return ntp_to_system_time(self.dest_timestamp)

class NTPClient(object):
    """NTP client session."""
    __slots__ = ()

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'Constructor.'
        pass

    def request(self, host, version=2, port='ntp', timeout=5):
        if False:
            i = 10
            return i + 15
        'Query a NTP server.\n\n        Parameters:\n        host    -- server name/address\n        version -- NTP version to use\n        port    -- server port\n        timeout -- timeout on socket operations\n\n        Returns:\n        NTPStats object\n        '
        addrinfo = socket.getaddrinfo(host, port)[0]
        (family, sockaddr) = (addrinfo[0], addrinfo[4])
        s = socket.socket(family, socket.SOCK_DGRAM)
        try:
            s.settimeout(timeout)
            query_packet = NTPPacket(mode=3, version=version, tx_timestamp=system_to_ntp_time(time.time()))
            s.sendto(query_packet.to_data(), sockaddr)
            src_addr = (None,)
            while src_addr[0] != sockaddr[0]:
                (response_packet, src_addr) = s.recvfrom(256)
            dest_timestamp = system_to_ntp_time(time.time())
        except socket.timeout:
            raise NTPException('No response received from %s.' % host)
        finally:
            s.close()
        stats = NTPStats()
        stats.from_data(response_packet)
        stats.dest_timestamp = dest_timestamp
        return stats

def _to_int(timestamp):
    if False:
        while True:
            i = 10
    'Return the integral part of a timestamp.\n\n    Parameters:\n    timestamp -- NTP timestamp\n\n    Retuns:\n    integral part\n    '
    return int(timestamp)

def _to_frac(timestamp, n=32):
    if False:
        print('Hello World!')
    'Return the fractional part of a timestamp.\n\n    Parameters:\n    timestamp -- NTP timestamp\n    n         -- number of bits of the fractional part\n\n    Retuns:\n    fractional part\n    '
    return int(abs(timestamp - _to_int(timestamp)) * 2 ** n)

def _to_time(integ, frac, n=32):
    if False:
        print('Hello World!')
    'Return a timestamp from an integral and fractional part.\n\n    Parameters:\n    integ -- integral part\n    frac  -- fractional part\n    n     -- number of bits of the fractional part\n\n    Retuns:\n    timestamp\n    '
    return integ + float(frac) / 2 ** n

def ntp_to_system_time(timestamp):
    if False:
        return 10
    'Convert a NTP time to system time.\n\n    Parameters:\n    timestamp -- timestamp in NTP time\n\n    Returns:\n    corresponding system time\n    '
    return timestamp - NTP.NTP_DELTA

def system_to_ntp_time(timestamp):
    if False:
        i = 10
        return i + 15
    'Convert a system time to a NTP time.\n\n    Parameters:\n    timestamp -- timestamp in system time\n\n    Returns:\n    corresponding NTP time\n    '
    return timestamp + NTP.NTP_DELTA

def leap_to_text(leap):
    if False:
        print('Hello World!')
    'Convert a leap indicator to text.\n\n    Parameters:\n    leap -- leap indicator value\n\n    Returns:\n    corresponding message\n\n    Raises:\n    NTPException -- in case of invalid leap indicator\n    '
    if leap in NTP.LEAP_TABLE:
        return NTP.LEAP_TABLE[leap]
    else:
        raise NTPException('Invalid leap indicator.')

def mode_to_text(mode):
    if False:
        for i in range(10):
            print('nop')
    'Convert a NTP mode value to text.\n\n    Parameters:\n    mode -- NTP mode\n\n    Returns:\n    corresponding message\n\n    Raises:\n    NTPException -- in case of invalid mode\n    '
    if mode in NTP.MODE_TABLE:
        return NTP.MODE_TABLE[mode]
    else:
        raise NTPException('Invalid mode.')

def stratum_to_text(stratum):
    if False:
        i = 10
        return i + 15
    'Convert a stratum value to text.\n\n    Parameters:\n    stratum -- NTP stratum\n\n    Returns:\n    corresponding message\n\n    Raises:\n    NTPException -- in case of invalid stratum\n    '
    if stratum in NTP.STRATUM_TABLE:
        return NTP.STRATUM_TABLE[stratum] % stratum
    elif 1 < stratum < 16:
        return 'secondary reference (%s)' % stratum
    elif stratum == 16:
        return 'unsynchronized (%s)' % stratum
    else:
        raise NTPException('Invalid stratum or reserved.')

def ref_id_to_text(ref_id, stratum=2):
    if False:
        while True:
            i = 10
    'Convert a reference clock identifier to text according to its stratum.\n\n    Parameters:\n    ref_id  -- reference clock indentifier\n    stratum -- NTP stratum\n\n    Returns:\n    corresponding message\n\n    Raises:\n    NTPException -- in case of invalid stratum\n    '
    fields = (ref_id >> 24 & 255, ref_id >> 16 & 255, ref_id >> 8 & 255, ref_id & 255)
    if 0 <= stratum <= 1:
        text = '%c%c%c%c' % fields
        if text in NTP.REF_ID_TABLE:
            return NTP.REF_ID_TABLE[text]
        else:
            return "Unidentified reference source '%s'" % text
    elif 2 <= stratum < 255:
        return '%d.%d.%d.%d' % fields
    else:
        raise NTPException('Invalid stratum.')