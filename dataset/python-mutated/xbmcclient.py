"""
Implementation of XBMC's UDP based input system.

A set of classes that abstract the various packets that the event server
currently supports. In addition, there's also a class, XBMCClient, that
provides functions that sends the various packets. Use XBMCClient if you
don't need complete control over packet structure.

The basic workflow involves:

1. Send a HELO packet
2. Send x number of valid packets
3. Send a BYE packet

IMPORTANT NOTE ABOUT TIMEOUTS:
A client is considered to be timed out if XBMC doesn't received a packet
at least once every 60 seconds. To "ping" XBMC with an empty packet use
PacketPING or XBMCClient.ping(). See the documentation for details.
"""
from __future__ import unicode_literals, print_function, absolute_import, division
__author__ = 'd4rk@xbmc.org'
__version__ = '0.1.0'
import sys
if sys.version_info.major == 2:
    str = unicode
from struct import pack
from socket import socket, AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_BROADCAST
import time
MAX_PACKET_SIZE = 1024
HEADER_SIZE = 32
MAX_PAYLOAD_SIZE = MAX_PACKET_SIZE - HEADER_SIZE
UNIQUE_IDENTIFICATION = int(time.time())
PT_HELO = 1
PT_BYE = 2
PT_BUTTON = 3
PT_MOUSE = 4
PT_PING = 5
PT_BROADCAST = 6
PT_NOTIFICATION = 7
PT_BLOB = 8
PT_LOG = 9
PT_ACTION = 10
PT_DEBUG = 255
ICON_NONE = 0
ICON_JPEG = 1
ICON_PNG = 2
ICON_GIF = 3
BT_USE_NAME = 1
BT_DOWN = 2
BT_UP = 4
BT_USE_AMOUNT = 8
BT_QUEUE = 16
BT_NO_REPEAT = 32
BT_VKEY = 64
BT_AXIS = 128
BT_AXISSINGLE = 256
MS_ABSOLUTE = 1
LOGDEBUG = 0
LOGINFO = 1
LOGWARNING = 2
LOGERROR = 3
LOGFATAL = 4
LOGNONE = 5
ACTION_EXECBUILTIN = 1
ACTION_BUTTON = 2

def format_string(msg):
    if False:
        for i in range(10):
            print('nop')
    ' '
    return msg.encode('utf-8') + b'\x00'

def format_uint32(num):
    if False:
        return 10
    ' '
    return pack('!I', num)

def format_uint16(num):
    if False:
        i = 10
        return i + 15
    ' '
    if num < 0:
        num = 0
    elif num > 65535:
        num = 65535
    return pack('!H', num)

class Packet:
    """Base class that implements a single event packet.

     - Generic packet structure (maximum 1024 bytes per packet)
     - Header is 32 bytes long, so 992 bytes available for payload
     - large payloads can be split into multiple packets using H4 and H5
       H5 should contain total no. of packets in such a case
     - H6 contains length of P1, which is limited to 992 bytes
     - if H5 is 0 or 1, then H4 will be ignored (single packet msg)
     - H7 must be set to zeros for now

         -----------------------------
         | -H1 Signature ("XBMC")    | - 4  x CHAR                4B
         | -H2 Version (eg. 2.0)     | - 2  x UNSIGNED CHAR       2B
         | -H3 PacketType            | - 1  x UNSIGNED SHORT      2B
         | -H4 Sequence number       | - 1  x UNSIGNED LONG       4B
         | -H5 No. of packets in msg | - 1  x UNSIGNED LONG       4B
         | -H7 Client's unique token | - 1  x UNSIGNED LONG       4B
         | -H8 Reserved              | - 10 x UNSIGNED CHAR      10B
         |---------------------------|
         | -P1 payload               | -
         -----------------------------
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.sig = b'XBMC'
        self.minver = 0
        self.majver = 2
        self.seq = 1
        self.maxseq = 1
        self.payloadsize = 0
        self.uid = UNIQUE_IDENTIFICATION
        self.reserved = b'\x00' * 10
        self.payload = b''

    def append_payload(self, blob):
        if False:
            while True:
                i = 10
        'Append to existing payload\n\n        Arguments:\n        blob -- binary data to append to the current payload\n        '
        if isinstance(blob, str):
            blob = blob.encode()
        self.set_payload(self.payload + blob)

    def set_payload(self, payload):
        if False:
            return 10
        'Set the payload for this packet\n\n        Arguments:\n        payload -- binary data that contains the payload\n        '
        if isinstance(payload, str):
            payload = payload.encode()
        self.payload = payload
        self.payloadsize = len(self.payload)
        self.maxseq = int((self.payloadsize + (MAX_PAYLOAD_SIZE - 1)) / MAX_PAYLOAD_SIZE)

    def num_packets(self):
        if False:
            print('Hello World!')
        ' Return the number of packets required for payload '
        return self.maxseq

    def get_header(self, packettype=-1, seq=1, maxseq=1, payload_size=0):
        if False:
            for i in range(10):
                print('nop')
        'Construct a header and return as string\n\n        Keyword arguments:\n        packettype -- valid packet types are PT_HELO, PT_BYE, PT_BUTTON,\n                      PT_MOUSE, PT_PING, PT_BORADCAST, PT_NOTIFICATION,\n                      PT_BLOB, PT_DEBUG\n        seq -- the sequence of this packet for a multi packet message\n               (default 1)\n        maxseq -- the total number of packets for a multi packet message\n                  (default 1)\n        payload_size -- the size of the payload of this packet (default 0)\n        '
        if packettype < 0:
            packettype = self.packettype
        header = self.sig
        header += chr(self.majver).encode()
        header += chr(self.minver).encode()
        header += format_uint16(packettype)
        header += format_uint32(seq)
        header += format_uint32(maxseq)
        header += format_uint16(payload_size)
        header += format_uint32(self.uid)
        header += self.reserved
        return header

    def get_payload_size(self, seq):
        if False:
            return 10
        'Returns the calculated payload size for the particular packet\n\n        Arguments:\n        seq -- the sequence number\n        '
        if self.maxseq == 1:
            return self.payloadsize
        if seq < self.maxseq:
            return MAX_PAYLOAD_SIZE
        return self.payloadsize % MAX_PAYLOAD_SIZE

    def get_udp_message(self, packetnum=1):
        if False:
            for i in range(10):
                print('nop')
        'Construct the UDP message for the specified packetnum and return\n        as string\n\n        Keyword arguments:\n        packetnum -- the packet no. for which to construct the message\n                     (default 1)\n        '
        if packetnum > self.num_packets() or packetnum < 1:
            return b''
        header = b''
        if packetnum == 1:
            header = self.get_header(self.packettype, packetnum, self.maxseq, self.get_payload_size(packetnum))
        else:
            header = self.get_header(PT_BLOB, packetnum, self.maxseq, self.get_payload_size(packetnum))
        payload = self.payload[(packetnum - 1) * MAX_PAYLOAD_SIZE:(packetnum - 1) * MAX_PAYLOAD_SIZE + self.get_payload_size(packetnum)]
        return header + payload

    def send(self, sock, addr, uid=UNIQUE_IDENTIFICATION):
        if False:
            while True:
                i = 10
        'Send the entire message to the specified socket and address.\n\n        Arguments:\n        sock -- datagram socket object (socket.socket)\n        addr -- address, port pair (eg: ("127.0.0.1", 9777) )\n        uid  -- unique identification\n        '
        self.uid = uid
        for a in range(0, self.num_packets()):
            sock.sendto(self.get_udp_message(a + 1), addr)

class PacketHELO(Packet):
    """A HELO packet

    A HELO packet establishes a valid connection to XBMC. It is the
    first packet that should be sent.
    """

    def __init__(self, devicename=None, icon_type=ICON_NONE, icon_file=None):
        if False:
            print('Hello World!')
        '\n        Keyword arguments:\n        devicename -- the string that identifies the client\n        icon_type -- one of ICON_NONE, ICON_JPEG, ICON_PNG, ICON_GIF\n        icon_file -- location of icon file with respect to current working\n                     directory if icon_type is not ICON_NONE\n        '
        Packet.__init__(self)
        self.packettype = PT_HELO
        self.icontype = icon_type
        self.set_payload(format_string(devicename)[0:128])
        self.append_payload(chr(icon_type))
        self.append_payload(format_uint16(0))
        self.append_payload(format_uint32(0))
        self.append_payload(format_uint32(0))
        if icon_type != ICON_NONE and icon_file:
            with open(icon_file, 'rb') as f:
                self.append_payload(f.read())

class PacketNOTIFICATION(Packet):
    """A NOTIFICATION packet

    This packet displays a notification window in XBMC. It can contain
    a caption, a message and an icon.
    """

    def __init__(self, title, message, icon_type=ICON_NONE, icon_file=None):
        if False:
            return 10
        '\n        Keyword arguments:\n        title -- the notification caption / title\n        message -- the main text of the notification\n        icon_type -- one of ICON_NONE, ICON_JPEG, ICON_PNG, ICON_GIF\n        icon_file -- location of icon file with respect to current working\n                     directory if icon_type is not ICON_NONE\n        '
        Packet.__init__(self)
        self.packettype = PT_NOTIFICATION
        self.title = title
        self.message = message
        self.set_payload(format_string(title))
        self.append_payload(format_string(message))
        self.append_payload(chr(icon_type))
        self.append_payload(format_uint32(0))
        if icon_type != ICON_NONE and icon_file:
            with open(icon_file, 'rb') as f:
                self.append_payload(f.read())

class PacketBUTTON(Packet):
    """A BUTTON packet

    A button packet send a key press or release event to XBMC
    """

    def __init__(self, code=0, repeat=1, down=1, queue=0, map_name='', button_name='', amount=0, axis=0):
        if False:
            return 10
        '\n        Keyword arguments:\n        code -- raw button code (default: 0)\n        repeat -- this key press should repeat until released (default: 1)\n                  Note that queued pressed cannot repeat.\n        down -- if this is 1, it implies a press event, 0 implies a release\n                event. (default: 1)\n        queue -- a queued key press means that the button event is\n                 executed just once after which the next key press is\n                 processed. It can be used for macros. Currently there\n                 is no support for time delays between queued presses.\n                 (default: 0)\n        map_name -- a combination of map_name and button_name refers to a\n                    mapping in the user\'s Keymap.xml or Lircmap.xml.\n                    map_name can be one of the following:\n                    "KB" => standard keyboard map ( <keyboard> section )\n                    "XG" => xbox gamepad map ( <gamepad> section )\n                    "R1" => xbox remote map ( <remote> section )\n                    "R2" => xbox universal remote map ( <universalremote>\n                            section )\n                    "LI:devicename" => LIRC remote map where \'devicename\' is the\n                    actual device\'s name\n        button_name -- a button name defined in the map specified in map_name.\n                       For example, if map_name is "KB" referring to the\n                       <keyboard> section in Keymap.xml then, valid\n                       button_names include "printscreen", "minus", "x", etc.\n        amount -- unimplemented for now; in the future it will be used for\n                  specifying magnitude of analog key press events\n        '
        Packet.__init__(self)
        self.flags = 0
        self.packettype = PT_BUTTON
        if type(code) == str:
            code = ord(code)
        if not (map_name and button_name):
            self.code = code
        else:
            self.flags |= BT_USE_NAME
            self.code = 0
        if amount != None:
            self.flags |= BT_USE_AMOUNT
            self.amount = int(amount)
        else:
            self.amount = 0
        if down:
            self.flags |= BT_DOWN
        else:
            self.flags |= BT_UP
        if not repeat:
            self.flags |= BT_NO_REPEAT
        if queue:
            self.flags |= BT_QUEUE
        if axis == 1:
            self.flags |= BT_AXISSINGLE
        elif axis == 2:
            self.flags |= BT_AXIS
        self.set_payload(format_uint16(self.code))
        self.append_payload(format_uint16(self.flags))
        self.append_payload(format_uint16(self.amount))
        self.append_payload(format_string(map_name))
        self.append_payload(format_string(button_name))

class PacketMOUSE(Packet):
    """A MOUSE packet

    A MOUSE packets sets the mouse position in XBMC
    """

    def __init__(self, x, y):
        if False:
            while True:
                i = 10
        '\n        Arguments:\n        x -- horizontal position ranging from 0 to 65535\n        y -- vertical position ranging from 0 to 65535\n\n        The range will be mapped to the screen width and height in XBMC\n        '
        Packet.__init__(self)
        self.packettype = PT_MOUSE
        self.flags = MS_ABSOLUTE
        self.append_payload(chr(self.flags))
        self.append_payload(format_uint16(x))
        self.append_payload(format_uint16(y))

class PacketBYE(Packet):
    """A BYE packet

    A BYE packet terminates the connection to XBMC.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        Packet.__init__(self)
        self.packettype = PT_BYE

class PacketPING(Packet):
    """A PING packet

    A PING packet tells XBMC that the client is still alive. All valid
    packets act as ping (not just this one). A client needs to ping
    XBMC at least once in 60 seconds or it will time out.
    """

    def __init__(self):
        if False:
            return 10
        Packet.__init__(self)
        self.packettype = PT_PING

class PacketLOG(Packet):
    """A LOG packet

    A LOG packet tells XBMC to log the message to xbmc.log with the loglevel as specified.
    """

    def __init__(self, loglevel=0, logmessage='', autoprint=True):
        if False:
            print('Hello World!')
        '\n        Keyword arguments:\n        loglevel -- the loglevel, follows XBMC standard.\n        logmessage -- the message to log\n        autoprint -- if the logmessage should automatically be printed to stdout\n        '
        Packet.__init__(self)
        self.packettype = PT_LOG
        self.append_payload(chr(loglevel))
        self.append_payload(format_string(logmessage))
        if autoprint:
            print(logmessage)

class PacketACTION(Packet):
    """An ACTION packet

    An ACTION packet tells XBMC to do the action specified, based on the type it knows were it needs to be sent.
    The idea is that this will be as in scripting/skining and keymapping, just triggered from afar.
    """

    def __init__(self, actionmessage='', actiontype=ACTION_EXECBUILTIN):
        if False:
            i = 10
            return i + 15
        '\n        Keyword arguments:\n        loglevel -- the loglevel, follows XBMC standard.\n        logmessage -- the message to log\n        autoprint -- if the logmessage should automatically be printed to stdout\n        '
        Packet.__init__(self)
        self.packettype = PT_ACTION
        self.append_payload(chr(actiontype))
        self.append_payload(format_string(actionmessage))

class XBMCClient:
    """An XBMC event client"""

    def __init__(self, name='', icon_file=None, broadcast=False, uid=UNIQUE_IDENTIFICATION, ip='127.0.0.1'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Keyword arguments:\n        name -- Name of the client\n        icon_file -- location of an icon file, if any (png, jpg or gif)\n        uid  -- unique identification\n        '
        self.name = str(name)
        self.icon_file = icon_file
        self.icon_type = self._get_icon_type(icon_file)
        self.ip = ip
        self.port = 9777
        self.sock = socket(AF_INET, SOCK_DGRAM)
        if broadcast:
            self.sock.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        self.uid = uid

    def connect(self, ip=None, port=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize connection to XBMC\n        ip -- IP Address of XBMC\n        port -- port that the event server on XBMC is listening on\n        '
        if ip:
            self.ip = ip
        if port:
            self.port = int(port)
        self.addr = (self.ip, self.port)
        packet = PacketHELO(self.name, self.icon_type, self.icon_file)
        packet.send(self.sock, self.addr, self.uid)

    def close(self):
        if False:
            while True:
                i = 10
        'Close the current connection'
        packet = PacketBYE()
        packet.send(self.sock, self.addr, self.uid)

    def ping(self):
        if False:
            for i in range(10):
                print('nop')
        'Send a PING packet'
        packet = PacketPING()
        packet.send(self.sock, self.addr, self.uid)

    def send_notification(self, title='', message='', icon_file=None):
        if False:
            print('Hello World!')
        'Send a notification to the connected XBMC\n        Keyword Arguments:\n        title -- The title/heading for the notification\n        message -- The message to be displayed\n        icon_file -- location of an icon file, if any (png, jpg, gif)\n        '
        self.connect()
        packet = PacketNOTIFICATION(title, message, self._get_icon_type(icon_file), icon_file)
        packet.send(self.sock, self.addr, self.uid)

    def send_keyboard_button(self, button=None):
        if False:
            print('Hello World!')
        'Send a keyboard event to XBMC\n        Keyword Arguments:\n        button -- name of the keyboard button to send (same as in Keymap.xml)\n        '
        if not button:
            return
        self.send_button(map='KB', button=button)

    def send_remote_button(self, button=None):
        if False:
            return 10
        'Send a remote control event to XBMC\n        Keyword Arguments:\n        button -- name of the remote control button to send (same as in Keymap.xml)\n        '
        if not button:
            return
        self.send_button(map='R1', button=button)

    def release_button(self):
        if False:
            i = 10
            return i + 15
        'Release all buttons'
        packet = PacketBUTTON(code=1, down=0)
        packet.send(self.sock, self.addr, self.uid)

    def send_button(self, map='', button='', amount=0):
        if False:
            return 10
        'Send a button event to XBMC\n        Keyword arguments:\n        map -- a combination of map_name and button_name refers to a\n               mapping in the user\'s Keymap.xml or Lircmap.xml.\n               map_name can be one of the following:\n                   "KB" => standard keyboard map ( <keyboard> section )\n                   "XG" => xbox gamepad map ( <gamepad> section )\n                   "R1" => xbox remote map ( <remote> section )\n                   "R2" => xbox universal remote map ( <universalremote>\n                           section )\n                   "LI:devicename" => LIRC remote map where \'devicename\' is the\n                                      actual device\'s name\n        button -- a button name defined in the map specified in map, above.\n                  For example, if map is "KB" referring to the <keyboard>\n                  section in Keymap.xml then, valid buttons include\n                  "printscreen", "minus", "x", etc.\n        '
        packet = PacketBUTTON(map_name=str(map), button_name=str(button), amount=amount)
        packet.send(self.sock, self.addr, self.uid)

    def send_button_state(self, map='', button='', amount=0, down=0, axis=0):
        if False:
            return 10
        'Send a button event to XBMC\n        Keyword arguments:\n        map -- a combination of map_name and button_name refers to a\n               mapping in the user\'s Keymap.xml or Lircmap.xml.\n               map_name can be one of the following:\n                   "KB" => standard keyboard map ( <keyboard> section )\n                   "XG" => xbox gamepad map ( <gamepad> section )\n                   "R1" => xbox remote map ( <remote> section )\n                   "R2" => xbox universal remote map ( <universalremote>\n                           section )\n                   "LI:devicename" => LIRC remote map where \'devicename\' is the\n                                      actual device\'s name\n        button -- a button name defined in the map specified in map, above.\n                  For example, if map is "KB" referring to the <keyboard>\n                  section in Keymap.xml then, valid buttons include\n                  "printscreen", "minus", "x", etc.\n        '
        if axis:
            down = int(amount != 0)
        packet = PacketBUTTON(map_name=str(map), button_name=str(button), amount=amount, down=down, queue=1, axis=axis)
        packet.send(self.sock, self.addr, self.uid)

    def send_mouse_position(self, x=0, y=0):
        if False:
            print('Hello World!')
        "Send a mouse event to XBMC\n        Keywords Arguments:\n        x -- absolute x position of mouse ranging from 0 to 65535\n             which maps to the entire screen width\n        y -- same a 'x' but relates to the screen height\n        "
        packet = PacketMOUSE(int(x), int(y))
        packet.send(self.sock, self.addr, self.uid)

    def send_log(self, loglevel=0, logmessage='', autoprint=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Keyword arguments:\n        loglevel -- the loglevel, follows XBMC standard.\n        logmessage -- the message to log\n        autoprint -- if the logmessage should automatically be printed to stdout\n        '
        packet = PacketLOG(loglevel, logmessage)
        packet.send(self.sock, self.addr, self.uid)

    def send_action(self, actionmessage='', actiontype=ACTION_EXECBUILTIN):
        if False:
            for i in range(10):
                print('nop')
        '\n        Keyword arguments:\n        actionmessage -- the ActionString\n        actiontype -- The ActionType the ActionString should be sent to.\n        '
        packet = PacketACTION(actionmessage, actiontype)
        packet.send(self.sock, self.addr, self.uid)

    def _get_icon_type(self, icon_file):
        if False:
            for i in range(10):
                print('nop')
        if icon_file:
            if icon_file.lower()[-3:] == 'png':
                return ICON_PNG
            elif icon_file.lower()[-3:] == 'gif':
                return ICON_GIF
            elif icon_file.lower()[-3:] == 'jpg':
                return ICON_JPEG
        return ICON_NONE