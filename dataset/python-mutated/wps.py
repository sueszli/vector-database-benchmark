import array
import struct
from impacket.ImpactPacket import array_tobytes
from impacket.helper import ProtocolPacket, Byte, Bit
from functools import reduce

class ArrayBuilder(object):

    def from_ary(self, ary):
        if False:
            print('Hello World!')
        return ary

    def to_ary(self, value):
        if False:
            print('Hello World!')
        return array.array('B', value)

class ByteBuilder(object):

    def from_ary(self, ary):
        if False:
            return 10
        return ary[0]

    def to_ary(self, value):
        if False:
            print('Hello World!')
        return array.array('B', [value])

class StringBuilder(object):

    def from_ary(self, ary):
        if False:
            print('Hello World!')
        return array_tobytes(ary)

    def to_ary(self, value):
        if False:
            while True:
                i = 10
        return array.array('B', value)

class NumBuilder(object):
    """Converts back and forth between arrays and numbers in network byte-order"""

    def __init__(self, size):
        if False:
            for i in range(10):
                print('nop')
        'size: number of bytes in the field'
        self.size = size

    def from_ary(self, ary):
        if False:
            while True:
                i = 10
        if len(ary) != self.size:
            raise Exception('Expected %s size but got %s' % (self.size, len(ary)))
        return reduce(lambda ac, x: ac * 256 + x, ary, 0)

    def to_ary(self, value0):
        if False:
            i = 10
            return i + 15
        value = value0
        rv = array.array('B')
        for _ in range(self.size):
            (value, mod) = divmod(value, 256)
            rv.append(mod)
        if value != 0:
            raise Exception('%s is too big. Max size: %s' % (value0, self.size))
        rv.reverse()
        return rv

class TLVContainer(object):

    def builder(self, kind):
        if False:
            print('Hello World!')
        return self.builders.get(kind, self.default_builder)

    def from_ary(self, ary):
        if False:
            print('Hello World!')
        i = 0
        while i < len(ary):
            kind = self.ary2n(ary, i)
            length = self.ary2n(ary, i + 2)
            i += 4
            value = ary[i:i + length]
            self.elems.append((kind, value))
            i += length
        return self

    def __init__(self, builders, default_builder=ArrayBuilder(), descs=None):
        if False:
            while True:
                i = 10
        self.builders = builders
        self.default_builder = default_builder
        self.elems = []
        self.descs = descs or {}

    def append(self, kind, value):
        if False:
            return 10
        self.elems.append((kind, self.builder(kind).to_ary(value)))

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return ((k, self.builder(k).from_ary(v)) for (k, v) in self.elems)

    def all(self, kind):
        if False:
            return 10
        return [e[1] for e in self if e[0] == kind]

    def __contains__(self, kind):
        if False:
            for i in range(10):
                print('nop')
        return len(self.all(kind)) != 0

    def first(self, kind):
        if False:
            print('Hello World!')
        return self.all(kind)[0]

    def to_ary(self):
        if False:
            print('Hello World!')
        ary = array.array('B')
        for (k, v) in self.elems:
            ary.extend(self.n2ary(k))
            ary.extend(self.n2ary(len(v)))
            ary.extend(v)
        return ary

    def get_packet(self):
        if False:
            for i in range(10):
                print('nop')
        return array_tobytes(self.to_ary())

    def set_parent(self, my_parent):
        if False:
            while True:
                i = 10
        self.__parent = my_parent

    def parent(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__parent

    def n2ary(self, n):
        if False:
            i = 10
            return i + 15
        return array.array('B', struct.pack('>H', n))

    def ary2n(self, ary, i=0):
        if False:
            while True:
                i = 10
        return struct.unpack('>H', array_tobytes(ary[i:i + 2]))[0]

    def __repr__(self):
        if False:
            while True:
                i = 10

        def desc(kind):
            if False:
                return 10
            return self.descs[kind] if kind in self.descs else kind
        return '<TLVContainer %s>' % repr([(desc(k), self.builder(k).from_ary(v)) for (k, v) in self.elems])

    def child(self):
        if False:
            for i in range(10):
                print('nop')
        return None

class SCElem(object):
    AP_CHANNEL = 4097
    ASSOCIATION_STATE = 4098
    AUTHENTICATION_TYPE = 4099
    AUTHENTICATION_TYPE_FLAGS = 4100
    AUTHENTICATOR = 4101
    CONFIG_METHODS = 4104
    CONFIGURATION_ERROR = 4105
    CONFIRMATION_URL4 = 4106
    CONFIRMATION_URL6 = 4107
    CONNECTION_TYPE = 4108
    CONNECTION_TYPE_FLAGS = 4109
    CREDENTIAL = 4110
    DEVICE_NAME = 4113
    DEVICE_PASSWORD_ID = 4114
    E_HASH1 = 4116
    E_HASH2 = 4117
    E_SNONCE1 = 4118
    E_SNONCE2 = 4119
    ENCRYPTED_SETTINGS = 4120
    ENCRYPTION_TYPE = 4111
    ENCRYPTION_TYPE_FLAGS = 4112
    ENROLLEE_NONCE = 4122
    FEATURE_ID = 4123
    IDENTITY = 4124
    INDENTITY_PROOF = 4125
    KEY_WRAP_AUTHENTICATOR = 4126
    KEY_IDENTIFIER = 4127
    MAC_ADDRESS = 4128
    MANUFACTURER = 4129
    MESSAGE_TYPE = 4130
    MODEL_NAME = 4131
    MODEL_NUMBER = 4132
    NETWORK_INDEX = 4134
    NETWORK_KEY = 4135
    NETWORK_KEY_INDEX = 4136
    NEW_DEVICE_NAME = 4137
    NEW_PASSWORD = 4138
    OOB_DEVICE_PASSWORD = 4140
    OS_VERSION = 4141
    POWER_LEVEL = 4143
    PSK_CURRENT = 4144
    PSK_MAX = 4145
    PUBLIC_KEY = 4146
    RADIO_ENABLED = 4147
    REBOOT = 4148
    REGISTRAR_CURRENT = 4149
    REGISTRAR_ESTABLISHED = 4150
    REGISTRAR_LIST = 4151
    REGISTRAR_MAX = 4152
    REGISTRAR_NONCE = 4153
    REQUEST_TYPE = 4154
    RESPONSE_TYPE = 4155
    RF_BANDS = 4156
    R_HASH1 = 4157
    R_HASH2 = 4158
    R_SNONCE1 = 4159
    R_SNONCE2 = 4160
    SELECTED_REGISTRAR = 4161
    SERIAL_NUMBER = 4162
    WPS_STATE = 4164
    SSID = 4165
    TOTAL_NETWORKS = 4166
    UUID_E = 4167
    UUID_R = 4168
    VENDOR_EXTENSION = 4169
    VERSION = 4170
    X_509_CERTIFICATE_REQUEST = 4171
    X_509_CERTIFICATE = 4172
    EAP_IDENTITY = 4173
    MESSAGE_COUNTER = 4174
    PUBLIC_KEY_HASH = 4175
    REKEY_KEY = 4176
    KEY_LIFETIME = 4177
    PERMITTED_CONFIG_METHODS = 4178
    SELECTED_REGISTRAR_CONFIG_METHODS = 4179
    PRIMARY_DEVICE_TYPE = 4180
    SECONDARY_DEVICE_TYPE_LIST = 4181
    PORTABLE_DEVICE = 4182
    AP_SETUP_LOCKED = 4183
    APPLICATION_EXTENSION = 4184
    EAP_TYPE = 4185
    INITIALIZATION_VECTOR = 4192
    KEY_PROVIDED_AUTOMATICALLY = 4193
    _802_1X_ENABLED = 4194
    APP_SESSION_KEY = 4195
    WEP_TRANSMIT_KEY = 4196

class MessageType(object):
    """Message types according to WPS 1.0h spec, section 11"""
    BEACON = 1
    PROBE_REQUEST = 2
    PROBE_RESPONSE = 3
    M1 = 4
    M2 = 5
    M2D = 6
    M3 = 7
    M4 = 8
    M5 = 9
    M6 = 10
    M7 = 11
    M8 = 12
    WSC_ACK = 13
    WSC_NACK = 14
    WSC_DONE = 15

class AuthTypeFlag(object):
    OPEN = 1
    WPAPSK = 2
    SHARED = 4
    WPA = 8
    WPA2 = 16
    WPA2PSK = 32
AuthTypeFlag_ALL = AuthTypeFlag.OPEN | AuthTypeFlag.WPAPSK | AuthTypeFlag.SHARED | AuthTypeFlag.WPA | AuthTypeFlag.WPA2 | AuthTypeFlag.WPA2PSK

class EncryptionTypeFlag(object):
    NONE = 1
    WEP = 2
    TKIP = 4
    AES = 8
EncryptionTypeFlag_ALL = EncryptionTypeFlag.NONE | EncryptionTypeFlag.WEP | EncryptionTypeFlag.TKIP | EncryptionTypeFlag.AES

class ConnectionTypeFlag(object):
    ESS = 1
    IBSS = 2

class ConfigMethod(object):
    USBA = 1
    ETHERNET = 2
    LABEL = 4
    DISPLAY = 8
    EXT_NFC_TOKEN = 16
    INT_NFC_TOKEN = 32
    NFC_INTERFACE = 64
    PUSHBUTTON = 128
    KEYPAD = 256

class OpCode(object):
    WSC_START = 1
    WSC_ACK = 2
    WSC_NACK = 3
    WSC_MSG = 4
    WSC_DONE = 5
    WSC_FRAG_ACK = 6

class AssocState(object):
    NOT_ASSOC = 0
    CONN_SUCCESS = 1
    CFG_FAILURE = 2
    FAILURE = (3,)
    IP_FAILURE = 4

class ConfigError(object):
    NO_ERROR = 0
    OOB_IFACE_READ_ERROR = 1
    DECRYPTION_CRC_FAILURE = 2
    _24_CHAN_NOT_SUPPORTED = 3
    _50_CHAN_NOT_SUPPORTED = 4
    SIGNAL_TOO_WEAK = 5
    NETWORK_AUTH_FAILURE = 6
    NETWORK_ASSOC_FAILURE = 7
    NO_DHCP_RESPONSE = 8
    FAILED_DHCP_CONFIG = 9
    IP_ADDR_CONFLICT = 10
    NO_CONN_TO_REGISTRAR = 11
    MULTIPLE_PBC_DETECTED = 12
    ROGUE_SUSPECTED = 13
    DEVICE_BUSY = 14
    SETUP_LOCKED = 15
    MSG_TIMEOUT = 16
    REG_SESS_TIMEOUT = 17
    DEV_PASSWORD_AUTH_FAILURE = 18

class DevicePasswordId(object):
    DEFAULT = 0
    USER_SPECIFIED = 1
    MACHINE_SPECIFIED = 2
    REKEY = 3
    PUSHBUTTON = 4
    REGISTRAR_SPECIFIED = 5

class WpsState(object):
    NOT_CONFIGURED = 1
    CONFIGURED = 2

class SimpleConfig(ProtocolPacket):
    """For now, it supports Simple configs with the bits more_fragments and length_field not set"""
    header_size = 2
    tail_size = 0
    op_code = Byte(0)
    flags = Byte(1)
    more_fragments = Bit(1, 0)
    length_field = Bit(1, 1)
    BUILDERS = {SCElem.CONNECTION_TYPE: ByteBuilder(), SCElem.CONNECTION_TYPE_FLAGS: ByteBuilder(), SCElem.VERSION: ByteBuilder(), SCElem.MESSAGE_TYPE: ByteBuilder(), SCElem.NETWORK_INDEX: ByteBuilder(), SCElem.NETWORK_KEY_INDEX: ByteBuilder(), SCElem.POWER_LEVEL: ByteBuilder(), SCElem.PSK_CURRENT: ByteBuilder(), SCElem.PSK_MAX: ByteBuilder(), SCElem.REGISTRAR_CURRENT: ByteBuilder(), SCElem.REGISTRAR_MAX: ByteBuilder(), SCElem.REQUEST_TYPE: ByteBuilder(), SCElem.RESPONSE_TYPE: ByteBuilder(), SCElem.RF_BANDS: ByteBuilder(), SCElem.WPS_STATE: ByteBuilder(), SCElem.TOTAL_NETWORKS: ByteBuilder(), SCElem.VERSION: ByteBuilder(), SCElem.WEP_TRANSMIT_KEY: ByteBuilder(), SCElem.CONFIRMATION_URL4: StringBuilder(), SCElem.CONFIRMATION_URL6: StringBuilder(), SCElem.DEVICE_NAME: StringBuilder(), SCElem.IDENTITY: StringBuilder(), SCElem.MANUFACTURER: StringBuilder(), SCElem.MODEL_NAME: StringBuilder(), SCElem.MODEL_NUMBER: StringBuilder(), SCElem.NEW_DEVICE_NAME: StringBuilder(), SCElem.NEW_PASSWORD: StringBuilder(), SCElem.SERIAL_NUMBER: StringBuilder(), SCElem.EAP_IDENTITY: StringBuilder(), SCElem.NETWORK_KEY: StringBuilder(), SCElem.AP_CHANNEL: NumBuilder(2), SCElem.ASSOCIATION_STATE: NumBuilder(2), SCElem.AUTHENTICATION_TYPE: NumBuilder(2), SCElem.AUTHENTICATION_TYPE_FLAGS: NumBuilder(2), SCElem.CONFIG_METHODS: NumBuilder(2), SCElem.CONFIGURATION_ERROR: NumBuilder(2), SCElem.DEVICE_PASSWORD_ID: NumBuilder(2), SCElem.ENCRYPTION_TYPE: NumBuilder(2), SCElem.ENCRYPTION_TYPE_FLAGS: NumBuilder(2), SCElem.MESSAGE_COUNTER: NumBuilder(8), SCElem.KEY_LIFETIME: NumBuilder(4), SCElem.PERMITTED_CONFIG_METHODS: NumBuilder(2), SCElem.SELECTED_REGISTRAR_CONFIG_METHODS: NumBuilder(2), SCElem.PUBLIC_KEY: NumBuilder(192)}

    @classmethod
    def build_tlv_container(cls):
        if False:
            for i in range(10):
                print('nop')
        return TLVContainer(builders=SimpleConfig.BUILDERS, descs=dict(((v, k) for (k, v) in SCElem.__dict__.items())))