"""Native Qt event filter.

This entire file is a giant WORKAROUND for https://bugreports.qt.io/browse/QTBUG-114334.
"""
from typing import Tuple, Union, cast, Optional
import enum
import ctypes
import ctypes.util
from qutebrowser.qt import sip, machinery
from qutebrowser.qt.core import QAbstractNativeEventFilter, QByteArray, qVersion
from qutebrowser.misc import objects
from qutebrowser.utils import log
_instance = None

class xcb_ge_generic_event_t(ctypes.Structure):
    """See https://xcb.freedesktop.org/manual/structxcb__ge__generic__event__t.html.

    Also used for xcb_generic_event_t as the structures overlap:
    https://xcb.freedesktop.org/manual/structxcb__generic__event__t.html
    """
    _fields_ = [('response_type', ctypes.c_uint8), ('extension', ctypes.c_uint8), ('sequence', ctypes.c_uint16), ('length', ctypes.c_uint32), ('event_type', ctypes.c_uint16), ('pad0', ctypes.c_uint8 * 22), ('full_sequence', ctypes.c_uint32)]
_XCB_GE_GENERIC = 35

class XcbInputOpcodes(enum.IntEnum):
    """https://xcb.freedesktop.org/manual/group__XCB__Input__API.html.

    NOTE: If adding anything new here, adjust _PROBLEMATIC_XINPUT_EVENTS below!
    """
    HIERARCHY = 11
    TOUCH_BEGIN = 18
    TOUCH_UPDATE = 19
    TOUCH_END = 20
    GESTURE_PINCH_BEGIN = 27
    GESTURE_PINCH_UPDATE = 28
    GESTURE_PINCH_END = 29
    GESTURE_SWIPE_BEGIN = 30
    GESTURE_SWIPE_UPDATE = 31
    GESTURE_SWIPE_END = 32
_PROBLEMATIC_XINPUT_EVENTS = set(XcbInputOpcodes) - {XcbInputOpcodes.HIERARCHY}

class xcb_query_extension_reply_t(ctypes.Structure):
    """https://xcb.freedesktop.org/manual/structxcb__query__extension__reply__t.html."""
    _fields_ = [('response_type', ctypes.c_uint8), ('pad0', ctypes.c_uint8), ('sequence', ctypes.c_uint16), ('length', ctypes.c_uint32), ('present', ctypes.c_uint8), ('major_opcode', ctypes.c_uint8), ('first_event', ctypes.c_uint8), ('first_error', ctypes.c_uint8)]
if machinery.IS_QT6:
    _PointerRetType = sip.voidptr
else:
    _PointerRetType = int

class NativeEventFilter(QAbstractNativeEventFilter):
    """Event filter for XCB messages to work around Qt 6.5.1 crash."""
    _PASS_EVENT_RET = (False, cast(_PointerRetType, 0))
    _FILTER_EVENT_RET = (True, cast(_PointerRetType, 0))

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        self._active = False
        xcb = ctypes.CDLL(ctypes.util.find_library('xcb'))
        xcb.xcb_connect.restype = ctypes.POINTER(ctypes.c_void_p)
        xcb.xcb_query_extension_reply.restype = ctypes.POINTER(xcb_query_extension_reply_t)
        conn = xcb.xcb_connect(None, None)
        assert conn
        try:
            assert not xcb.xcb_connection_has_error(conn)
            name = b'XInputExtension'
            cookie = xcb.xcb_query_extension(conn, len(name), name)
            reply = xcb.xcb_query_extension_reply(conn, cookie, None)
            assert reply
            if reply.contents.present:
                self.xinput_opcode = reply.contents.major_opcode
            else:
                self.xinput_opcode = None
        finally:
            xcb.xcb_disconnect(conn)

    def nativeEventFilter(self, evtype: Union[bytes, QByteArray], message: Optional[sip.voidptr]) -> Tuple[bool, _PointerRetType]:
        if False:
            return 10
        'Handle XCB events.'
        assert evtype == b'xcb_generic_event_t', evtype
        assert message is not None
        event = ctypes.cast(int(message), ctypes.POINTER(xcb_ge_generic_event_t)).contents
        if event.response_type == _XCB_GE_GENERIC and event.extension == self.xinput_opcode:
            if not self._active and event.event_type == XcbInputOpcodes.HIERARCHY:
                log.misc.warning('Got XInput HIERARCHY event, future swipe/pinch/touch events will be ignored to avoid a Qt 6.5.1 crash. Restart qutebrowser to make them work again.')
                self._active = True
            elif self._active and event.event_type in _PROBLEMATIC_XINPUT_EVENTS:
                name = XcbInputOpcodes(event.event_type).name
                log.misc.debug(f'Ignoring problematic XInput event {name}')
                return self._FILTER_EVENT_RET
        return self._PASS_EVENT_RET

def init() -> None:
    if False:
        while True:
            i = 10
    'Install the native event filter if needed.'
    global _instance
    platform = objects.qapp.platformName()
    qt_version = qVersion()
    log.misc.debug(f'Platform {platform}, Qt {qt_version}')
    if platform != 'xcb' or qt_version != '6.5.1':
        return
    log.misc.debug('Installing native event filter to work around Qt 6.5.1 crash')
    _instance = NativeEventFilter()
    objects.qapp.installNativeEventFilter(_instance)