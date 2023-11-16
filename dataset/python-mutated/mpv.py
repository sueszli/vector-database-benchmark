from ctypes import *
import ctypes.util
import threading
import os
import sys
from warnings import warn
from functools import partial, wraps
from contextlib import contextmanager
import collections
import re
import traceback
_env_name = 'MPV_DYLIB_PATH'
_nt_err_msg = 'Cannot find mpv-1.dll in your system %PATH%. One way to deal with this is to ship mpv-1.dll with your script and put the directory your script is in into %PATH% before "import mpv": os.environ["PATH"] = os.path.dirname(__file__) + os.pathsep + os.environ["PATH"] If mpv-1.dll is located elsewhere, you can add that path to os.environ["PATH"].'
_nix_err_msg = "Cannot find libmpv in the usual places. Depending on your distro, you may try installing an mpv-devel or mpv-libs package. If you have libmpv around but this script can't find it, consult the documentation for ctypes.util.find_library which this script uses to look up the library filename."
if os.name == 'nt':
    _default_mpv_dylib = 'mpv-1.dll'
    _err_msg = _nt_err_msg
    fs_enc = 'utf-8'
else:
    import locale
    (lc, enc) = locale.getlocale(locale.LC_NUMERIC)
    locale.setlocale(locale.LC_NUMERIC, 'C')
    _default_mpv_dylib = 'mpv'
    _err_msg = _nix_err_msg
    fs_enc = sys.getfilesystemencoding()
_user_mpv_dylib = os.environ.get(_env_name, None)
if _user_mpv_dylib is not None:
    backend = CDLL(_user_mpv_dylib)
else:
    _dll = ctypes.util.find_library(_default_mpv_dylib)
    if _dll is None:
        raise OSError(_err_msg)
    backend = CDLL(_dll)

class ShutdownError(SystemError):
    pass

class MpvHandle(c_void_p):
    pass

class MpvRenderCtxHandle(c_void_p):
    pass

class MpvOpenGLCbContext(c_void_p):
    pass

class PropertyUnavailableError(AttributeError):
    pass

class ErrorCode(object):
    """For documentation on these, see mpv's libmpv/client.h."""
    SUCCESS = 0
    EVENT_QUEUE_FULL = -1
    NOMEM = -2
    UNINITIALIZED = -3
    INVALID_PARAMETER = -4
    OPTION_NOT_FOUND = -5
    OPTION_FORMAT = -6
    OPTION_ERROR = -7
    PROPERTY_NOT_FOUND = -8
    PROPERTY_FORMAT = -9
    PROPERTY_UNAVAILABLE = -10
    PROPERTY_ERROR = -11
    COMMAND = -12
    LOADING_FAILED = -13
    AO_INIT_FAILED = -14
    VO_INIT_FAILED = -15
    NOTHING_TO_PLAY = -16
    UNKNOWN_FORMAT = -17
    UNSUPPORTED = -18
    NOT_IMPLEMENTED = -19
    GENERIC = -20
    EXCEPTION_DICT = {0: None, -1: lambda *a: MemoryError('mpv event queue full', *a), -2: lambda *a: MemoryError('mpv cannot allocate memory', *a), -3: lambda *a: ValueError('Uninitialized mpv handle used', *a), -4: lambda *a: ValueError('Invalid value for mpv parameter', *a), -5: lambda *a: AttributeError('mpv option does not exist', *a), -6: lambda *a: TypeError('Tried to set mpv option using wrong format', *a), -7: lambda *a: ValueError('Invalid value for mpv option', *a), -8: lambda *a: AttributeError('mpv property does not exist', *a), -9: lambda *a: TypeError('Tried to get/set mpv property using wrong format, or passed invalid value', *a), -10: lambda *a: PropertyUnavailableError('mpv property is not available', *a), -11: lambda *a: RuntimeError('Generic error getting or setting mpv property', *a), -12: lambda *a: SystemError('Error running mpv command', *a), -14: lambda *a: RuntimeError('Initializing the audio output failed', *a), -15: lambda *a: RuntimeError('Initializing the video output failed'), -16: lambda *a: RuntimeError('There was no audio or video data to play. This also happens if the file was recognized, but did not contain any audio or video streams, or no streams were selected.'), -17: lambda *a: RuntimeError('When trying to load the file, the file format could not be determined, or the file was too broken to open it'), -18: lambda *a: ValueError('Generic error for signaling that certain system requirements are not fulfilled'), -19: lambda *a: NotImplementedError('The API function which was called is a stub only'), -20: lambda *a: RuntimeError('Unspecified error')}

    @staticmethod
    def default_error_handler(ec, *args):
        if False:
            while True:
                i = 10
        return ValueError(_mpv_error_string(ec).decode('utf-8'), ec, *args)

    @classmethod
    def raise_for_ec(kls, ec, func, *args):
        if False:
            return 10
        ec = 0 if ec > 0 else ec
        ex = kls.EXCEPTION_DICT.get(ec, kls.default_error_handler)
        if ex:
            raise ex(ec, *args)
MpvGlGetProcAddressFn = CFUNCTYPE(c_void_p, c_void_p, c_char_p)

class MpvOpenGLInitParams(Structure):
    _fields_ = [('get_proc_address', MpvGlGetProcAddressFn), ('get_proc_address_ctx', c_void_p), ('extra_exts', c_void_p)]

    def __init__(self, get_proc_address):
        if False:
            for i in range(10):
                print('nop')
        self.get_proc_address = get_proc_address
        self.get_proc_address_ctx = None
        self.extra_exts = None

class MpvOpenGLFBO(Structure):
    _fields_ = [('fbo', c_int), ('w', c_int), ('h', c_int), ('internal_format', c_int)]

    def __init__(self, w, h, fbo=0, internal_format=0):
        if False:
            for i in range(10):
                print('nop')
        (self.w, self.h) = (w, h)
        self.fbo = fbo
        self.internal_format = internal_format

class MpvRenderFrameInfo(Structure):
    _fields_ = [('flags', c_int64), ('target_time', c_int64)]

    def as_dict(self):
        if False:
            return 10
        return {'flags': self.flags, 'target_time': self.target_time}

class MpvOpenGLDRMParams(Structure):
    _fields_ = [('fd', c_int), ('crtc_id', c_int), ('connector_id', c_int), ('atomic_request_ptr', c_void_p), ('render_fd', c_int)]

class MpvOpenGLDRMDrawSurfaceSize(Structure):
    _fields_ = [('width', c_int), ('height', c_int)]

class MpvOpenGLDRMParamsV2(Structure):
    _fields_ = [('fd', c_int), ('crtc_id', c_int), ('connector_id', c_int), ('atomic_request_ptr', c_void_p), ('render_fd', c_int)]

    def __init__(self, crtc_id, connector_id, atomic_request_ptr, fd=-1, render_fd=-1):
        if False:
            print('Hello World!')
        (self.crtc_id, self.connector_id) = (crtc_id, connector_id)
        self.atomic_request_ptr = atomic_request_ptr
        (self.fd, self.render_fd) = (fd, render_fd)

class MpvRenderParam(Structure):
    _fields_ = [('type_id', c_int), ('data', c_void_p)]
    TYPES = {'invalid': (0, None), 'api_type': (1, str), 'opengl_init_params': (2, MpvOpenGLInitParams), 'opengl_fbo': (3, MpvOpenGLFBO), 'flip_y': (4, bool), 'depth': (5, int), 'icc_profile': (6, bytes), 'ambient_light': (7, int), 'x11_display': (8, c_void_p), 'wl_display': (9, c_void_p), 'advanced_control': (10, bool), 'next_frame_info': (11, MpvRenderFrameInfo), 'block_for_target_time': (12, bool), 'skip_rendering': (13, bool), 'drm_display': (14, MpvOpenGLDRMParams), 'drm_draw_surface_size': (15, MpvOpenGLDRMDrawSurfaceSize), 'drm_display_v2': (16, MpvOpenGLDRMParamsV2)}

    def __init__(self, name, value=None):
        if False:
            return 10
        if name not in self.TYPES:
            raise ValueError('unknown render param type "{}"'.format(name))
        (self.type_id, cons) = self.TYPES[name]
        if cons is None:
            self.value = None
            self.data = c_void_p()
        elif cons is str:
            self.value = value
            self.data = cast(c_char_p(value.encode('utf-8')), c_void_p)
        elif cons is bytes:
            self.value = MpvByteArray(value)
            self.data = cast(pointer(self.value), c_void_p)
        elif cons is bool:
            self.value = c_int(int(bool(value)))
            self.data = cast(pointer(self.value), c_void_p)
        else:
            self.value = cons(**value)
            self.data = cast(pointer(self.value), c_void_p)

def kwargs_to_render_param_array(kwargs):
    if False:
        return 10
    t = MpvRenderParam * (len(kwargs) + 1)
    return t(*kwargs.items(), ('invalid', None))

class MpvFormat(c_int):
    NONE = 0
    STRING = 1
    OSD_STRING = 2
    FLAG = 3
    INT64 = 4
    DOUBLE = 5
    NODE = 6
    NODE_ARRAY = 7
    NODE_MAP = 8
    BYTE_ARRAY = 9

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self is other or self.value == other or self.value == int(other)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return ['NONE', 'STRING', 'OSD_STRING', 'FLAG', 'INT64', 'DOUBLE', 'NODE', 'NODE_ARRAY', 'NODE_MAP', 'BYTE_ARRAY'][self.value]

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.value

class MpvEventID(c_int):
    NONE = 0
    SHUTDOWN = 1
    LOG_MESSAGE = 2
    GET_PROPERTY_REPLY = 3
    SET_PROPERTY_REPLY = 4
    COMMAND_REPLY = 5
    START_FILE = 6
    END_FILE = 7
    FILE_LOADED = 8
    TRACKS_CHANGED = 9
    TRACK_SWITCHED = 10
    IDLE = 11
    PAUSE = 12
    UNPAUSE = 13
    TICK = 14
    SCRIPT_INPUT_DISPATCH = 15
    CLIENT_MESSAGE = 16
    VIDEO_RECONFIG = 17
    AUDIO_RECONFIG = 18
    METADATA_UPDATE = 19
    SEEK = 20
    PLAYBACK_RESTART = 21
    PROPERTY_CHANGE = 22
    CHAPTER_CHANGE = 23
    ANY = (SHUTDOWN, LOG_MESSAGE, GET_PROPERTY_REPLY, SET_PROPERTY_REPLY, COMMAND_REPLY, START_FILE, END_FILE, FILE_LOADED, TRACKS_CHANGED, TRACK_SWITCHED, IDLE, PAUSE, UNPAUSE, TICK, SCRIPT_INPUT_DISPATCH, CLIENT_MESSAGE, VIDEO_RECONFIG, AUDIO_RECONFIG, METADATA_UPDATE, SEEK, PLAYBACK_RESTART, PROPERTY_CHANGE, CHAPTER_CHANGE)

    def __repr__(self):
        if False:
            print('Hello World!')
        return ['NONE', 'SHUTDOWN', 'LOG_MESSAGE', 'GET_PROPERTY_REPLY', 'SET_PROPERTY_REPLY', 'COMMAND_REPLY', 'START_FILE', 'END_FILE', 'FILE_LOADED', 'TRACKS_CHANGED', 'TRACK_SWITCHED', 'IDLE', 'PAUSE', 'UNPAUSE', 'TICK', 'SCRIPT_INPUT_DISPATCH', 'CLIENT_MESSAGE', 'VIDEO_RECONFIG', 'AUDIO_RECONFIG', 'METADATA_UPDATE', 'SEEK', 'PLAYBACK_RESTART', 'PROPERTY_CHANGE', 'CHAPTER_CHANGE'][self.value]

    @classmethod
    def from_str(kls, s):
        if False:
            print('Hello World!')
        return getattr(kls, s.upper().replace('-', '_'))
identity_decoder = lambda b: b
strict_decoder = lambda b: b.decode('utf-8')

def lazy_decoder(b):
    if False:
        i = 10
        return i + 15
    try:
        return b.decode('utf-8')
    except UnicodeDecodeError:
        return b

class MpvNodeList(Structure):

    def array_value(self, decoder=identity_decoder):
        if False:
            return 10
        return [self.values[i].node_value(decoder) for i in range(self.num)]

    def dict_value(self, decoder=identity_decoder):
        if False:
            return 10
        return {self.keys[i].decode('utf-8'): self.values[i].node_value(decoder) for i in range(self.num)}

class MpvByteArray(Structure):
    _fields_ = [('data', c_void_p), ('size', c_size_t)]

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._value = value
        self.data = cast(c_char_p(value), c_void_p)
        self.size = len(value)

    def bytes_value(self):
        if False:
            return 10
        return cast(self.data, POINTER(c_char))[:self.size]

class MpvNode(Structure):

    def node_value(self, decoder=identity_decoder):
        if False:
            i = 10
            return i + 15
        return MpvNode.node_cast_value(self.val, self.format.value, decoder)

    @staticmethod
    def node_cast_value(v, fmt=MpvFormat.NODE, decoder=identity_decoder):
        if False:
            for i in range(10):
                print('nop')
        if fmt == MpvFormat.NONE:
            return None
        elif fmt == MpvFormat.STRING:
            return decoder(v.string)
        elif fmt == MpvFormat.OSD_STRING:
            return v.string.decode('utf-8')
        elif fmt == MpvFormat.FLAG:
            return bool(v.flag)
        elif fmt == MpvFormat.INT64:
            return v.int64
        elif fmt == MpvFormat.DOUBLE:
            return v.double
        else:
            if not v.node:
                return None
            if fmt == MpvFormat.NODE:
                return v.node.contents.node_value(decoder)
            elif fmt == MpvFormat.NODE_ARRAY:
                return v.list.contents.array_value(decoder)
            elif fmt == MpvFormat.NODE_MAP:
                return v.map.contents.dict_value(decoder)
            elif fmt == MpvFormat.BYTE_ARRAY:
                return v.byte_array.contents.bytes_value()
            else:
                raise TypeError('Unknown MPV node format {}. Please submit a bug report.'.format(fmt))

class MpvNodeUnion(Union):
    _fields_ = [('string', c_char_p), ('flag', c_int), ('int64', c_int64), ('double', c_double), ('node', POINTER(MpvNode)), ('list', POINTER(MpvNodeList)), ('map', POINTER(MpvNodeList)), ('byte_array', POINTER(MpvByteArray))]
MpvNode._fields_ = [('val', MpvNodeUnion), ('format', MpvFormat)]
MpvNodeList._fields_ = [('num', c_int), ('values', POINTER(MpvNode)), ('keys', POINTER(c_char_p))]

class MpvSubApi(c_int):
    MPV_SUB_API_OPENGL_CB = 1

class MpvEvent(Structure):
    _fields_ = [('event_id', MpvEventID), ('error', c_int), ('reply_userdata', c_ulonglong), ('data', c_void_p)]

    def as_dict(self, decoder=identity_decoder):
        if False:
            for i in range(10):
                print('nop')
        dtype = {MpvEventID.END_FILE: MpvEventEndFile, MpvEventID.PROPERTY_CHANGE: MpvEventProperty, MpvEventID.GET_PROPERTY_REPLY: MpvEventProperty, MpvEventID.LOG_MESSAGE: MpvEventLogMessage, MpvEventID.SCRIPT_INPUT_DISPATCH: MpvEventScriptInputDispatch, MpvEventID.CLIENT_MESSAGE: MpvEventClientMessage}.get(self.event_id.value, None)
        return {'event_id': self.event_id.value, 'error': self.error, 'reply_userdata': self.reply_userdata, 'event': cast(self.data, POINTER(dtype)).contents.as_dict(decoder=decoder) if dtype else None}

class MpvEventProperty(Structure):
    _fields_ = [('name', c_char_p), ('format', MpvFormat), ('data', MpvNodeUnion)]

    def as_dict(self, decoder=identity_decoder):
        if False:
            for i in range(10):
                print('nop')
        value = MpvNode.node_cast_value(self.data, self.format.value, decoder)
        return {'name': self.name.decode('utf-8'), 'format': self.format, 'data': self.data, 'value': value}

class MpvEventLogMessage(Structure):
    _fields_ = [('prefix', c_char_p), ('level', c_char_p), ('text', c_char_p)]

    def as_dict(self, decoder=identity_decoder):
        if False:
            while True:
                i = 10
        return {'prefix': self.prefix.decode('utf-8'), 'level': self.level.decode('utf-8'), 'text': decoder(self.text).rstrip()}

class MpvEventEndFile(Structure):
    _fields_ = [('reason', c_int), ('error', c_int)]
    EOF = 0
    RESTARTED = 1
    ABORTED = 2
    QUIT = 3
    ERROR = 4
    REDIRECT = 5

    @property
    def value(self):
        if False:
            for i in range(10):
                print('nop')
        return self.reason

    def as_dict(self, decoder=identity_decoder):
        if False:
            i = 10
            return i + 15
        return {'reason': self.reason, 'error': self.error}

class MpvEventScriptInputDispatch(Structure):
    _fields_ = [('arg0', c_int), ('type', c_char_p)]

    def as_dict(self, decoder=identity_decoder):
        if False:
            i = 10
            return i + 15
        pass

class MpvEventClientMessage(Structure):
    _fields_ = [('num_args', c_int), ('args', POINTER(c_char_p))]

    def as_dict(self, decoder=identity_decoder):
        if False:
            return 10
        return {'args': [self.args[i].decode('utf-8') for i in range(self.num_args)]}
StreamReadFn = CFUNCTYPE(c_int64, c_void_p, POINTER(c_char), c_uint64)
StreamSeekFn = CFUNCTYPE(c_int64, c_void_p, c_int64)
StreamSizeFn = CFUNCTYPE(c_int64, c_void_p)
StreamCloseFn = CFUNCTYPE(None, c_void_p)
StreamCancelFn = CFUNCTYPE(None, c_void_p)

class StreamCallbackInfo(Structure):
    _fields_ = [('cookie', c_void_p), ('read', StreamReadFn), ('seek', StreamSeekFn), ('size', StreamSizeFn), ('close', StreamCloseFn)]
StreamOpenFn = CFUNCTYPE(c_int, c_void_p, c_char_p, POINTER(StreamCallbackInfo))
WakeupCallback = CFUNCTYPE(None, c_void_p)
RenderUpdateFn = CFUNCTYPE(None, c_void_p)
OpenGlCbUpdateFn = CFUNCTYPE(None, c_void_p)
OpenGlCbGetProcAddrFn = CFUNCTYPE(c_void_p, c_void_p, c_char_p)

def _handle_func(name, args, restype, errcheck, ctx=MpvHandle, deprecated=False):
    if False:
        for i in range(10):
            print('nop')
    func = getattr(backend, name)
    func.argtypes = [ctx] + args if ctx else args
    if restype is not None:
        func.restype = restype
    if errcheck is not None:
        func.errcheck = errcheck
    if deprecated:

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                while True:
                    i = 10
            if not wrapper.warned:
                warn('Backend C api has been deprecated: ' + name, DeprecationWarning, stacklevel=2)
                wrapper.warned = True
            return func(*args, **kwargs)
        wrapper.warned = False
        globals()['_' + name] = wrapper
    else:
        globals()['_' + name] = func

def bytes_free_errcheck(res, func, *args):
    if False:
        while True:
            i = 10
    notnull_errcheck(res, func, *args)
    rv = cast(res, c_void_p).value
    _mpv_free(res)
    return rv

def notnull_errcheck(res, func, *args):
    if False:
        while True:
            i = 10
    if res is None:
        raise RuntimeError('Underspecified error in MPV when calling {} with args {!r}: NULL pointer returned.Please consult your local debugger.'.format(func.__name__, args))
    return res
ec_errcheck = ErrorCode.raise_for_ec

def _handle_gl_func(name, args=[], restype=None, deprecated=False):
    if False:
        for i in range(10):
            print('nop')
    _handle_func(name, args, restype, errcheck=None, ctx=MpvOpenGLCbContext, deprecated=deprecated)
backend.mpv_client_api_version.restype = c_ulong

def _mpv_client_api_version():
    if False:
        return 10
    ver = backend.mpv_client_api_version()
    return (ver >> 16, ver & 65535)
backend.mpv_free.argtypes = [c_void_p]
_mpv_free = backend.mpv_free
backend.mpv_free_node_contents.argtypes = [c_void_p]
_mpv_free_node_contents = backend.mpv_free_node_contents
backend.mpv_create.restype = MpvHandle
_mpv_create = backend.mpv_create
_handle_func('mpv_create_client', [c_char_p], MpvHandle, notnull_errcheck)
_handle_func('mpv_client_name', [], c_char_p, errcheck=None)
_handle_func('mpv_initialize', [], c_int, ec_errcheck)
_handle_func('mpv_destroy', [], None, errcheck=None)
_handle_func('mpv_terminate_destroy', [], None, errcheck=None)
_handle_func('mpv_load_config_file', [c_char_p], c_int, ec_errcheck)
_handle_func('mpv_get_time_us', [], c_ulonglong, errcheck=None)
_handle_func('mpv_set_option', [c_char_p, MpvFormat, c_void_p], c_int, ec_errcheck)
_handle_func('mpv_set_option_string', [c_char_p, c_char_p], c_int, ec_errcheck)
_handle_func('mpv_command', [POINTER(c_char_p)], c_int, ec_errcheck)
_handle_func('mpv_command_string', [c_char_p, c_char_p], c_int, ec_errcheck)
_handle_func('mpv_command_async', [c_ulonglong, POINTER(c_char_p)], c_int, ec_errcheck)
_handle_func('mpv_command_node', [POINTER(MpvNode), POINTER(MpvNode)], c_int, ec_errcheck)
_handle_func('mpv_command_async', [c_ulonglong, POINTER(MpvNode)], c_int, ec_errcheck)
_handle_func('mpv_set_property', [c_char_p, MpvFormat, c_void_p], c_int, ec_errcheck)
_handle_func('mpv_set_property_string', [c_char_p, c_char_p], c_int, ec_errcheck)
_handle_func('mpv_set_property_async', [c_ulonglong, c_char_p, MpvFormat, c_void_p], c_int, ec_errcheck)
_handle_func('mpv_get_property', [c_char_p, MpvFormat, c_void_p], c_int, ec_errcheck)
_handle_func('mpv_get_property_string', [c_char_p], c_void_p, bytes_free_errcheck)
_handle_func('mpv_get_property_osd_string', [c_char_p], c_void_p, bytes_free_errcheck)
_handle_func('mpv_get_property_async', [c_ulonglong, c_char_p, MpvFormat], c_int, ec_errcheck)
_handle_func('mpv_observe_property', [c_ulonglong, c_char_p, MpvFormat], c_int, ec_errcheck)
_handle_func('mpv_unobserve_property', [c_ulonglong], c_int, ec_errcheck)
_handle_func('mpv_event_name', [c_int], c_char_p, errcheck=None, ctx=None)
_handle_func('mpv_error_string', [c_int], c_char_p, errcheck=None, ctx=None)
_handle_func('mpv_request_event', [MpvEventID, c_int], c_int, ec_errcheck)
_handle_func('mpv_request_log_messages', [c_char_p], c_int, ec_errcheck)
_handle_func('mpv_wait_event', [c_double], POINTER(MpvEvent), errcheck=None)
_handle_func('mpv_wakeup', [], None, errcheck=None)
_handle_func('mpv_set_wakeup_callback', [WakeupCallback, c_void_p], None, errcheck=None)
_handle_func('mpv_get_wakeup_pipe', [], c_int, errcheck=None)
_handle_func('mpv_stream_cb_add_ro', [c_char_p, c_void_p, StreamOpenFn], c_int, ec_errcheck)
_handle_func('mpv_render_context_create', [MpvRenderCtxHandle, MpvHandle, POINTER(MpvRenderParam)], c_int, ec_errcheck, ctx=None)
_handle_func('mpv_render_context_set_parameter', [MpvRenderParam], c_int, ec_errcheck, ctx=MpvRenderCtxHandle)
_handle_func('mpv_render_context_get_info', [MpvRenderParam], c_int, ec_errcheck, ctx=MpvRenderCtxHandle)
_handle_func('mpv_render_context_set_update_callback', [RenderUpdateFn, c_void_p], None, errcheck=None, ctx=MpvRenderCtxHandle)
_handle_func('mpv_render_context_update', [], c_int64, errcheck=None, ctx=MpvRenderCtxHandle)
_handle_func('mpv_render_context_render', [POINTER(MpvRenderParam)], c_int, ec_errcheck, ctx=MpvRenderCtxHandle)
_handle_func('mpv_render_context_report_swap', [], None, errcheck=None, ctx=MpvRenderCtxHandle)
_handle_func('mpv_render_context_free', [], None, errcheck=None, ctx=MpvRenderCtxHandle)
if hasattr(backend, 'mpv_get_sub_api'):
    _handle_func('mpv_get_sub_api', [MpvSubApi], c_void_p, notnull_errcheck, deprecated=True)
    _handle_gl_func('mpv_opengl_cb_set_update_callback', [OpenGlCbUpdateFn, c_void_p], deprecated=True)
    _handle_gl_func('mpv_opengl_cb_init_gl', [c_char_p, OpenGlCbGetProcAddrFn, c_void_p], c_int, deprecated=True)
    _handle_gl_func('mpv_opengl_cb_draw', [c_int, c_int, c_int], c_int, deprecated=True)
    _handle_gl_func('mpv_opengl_cb_render', [c_int, c_int], c_int, deprecated=True)
    _handle_gl_func('mpv_opengl_cb_report_flip', [c_ulonglong], c_int, deprecated=True)
    _handle_gl_func('mpv_opengl_cb_uninit_gl', [], c_int, deprecated=True)

def _mpv_coax_proptype(value, proptype=str):
    if False:
        while True:
            i = 10
    'Intelligently coax the given python value into something that can be understood as a proptype property.'
    if type(value) is bytes:
        return value
    elif type(value) is bool:
        return b'yes' if value else b'no'
    elif proptype in (str, int, float):
        return str(proptype(value)).encode('utf-8')
    else:
        raise TypeError('Cannot coax value of type {} into property type {}'.format(type(value), proptype))

def _make_node_str_list(l):
    if False:
        while True:
            i = 10
    'Take a list of python objects and make a MPV string node array from it.\n    As an example, the python list ``l = [ "foo", 23, false ]`` will result in the following MPV node object::\n        struct mpv_node {\n            .format = MPV_NODE_ARRAY,\n            .u.list = *(struct mpv_node_array){\n                .num = len(l),\n                .keys = NULL,\n                .values = struct mpv_node[len(l)] {\n                    { .format = MPV_NODE_STRING, .u.string = l[0] },\n                    { .format = MPV_NODE_STRING, .u.string = l[1] },\n                    ...\n                }\n            }\n        }\n    '
    char_ps = [c_char_p(_mpv_coax_proptype(e, str)) for e in l]
    node_list = MpvNodeList(num=len(l), keys=None, values=(MpvNode * len(l))(*[MpvNode(format=MpvFormat.STRING, val=MpvNodeUnion(string=p)) for p in char_ps]))
    node = MpvNode(format=MpvFormat.NODE_ARRAY, val=MpvNodeUnion(list=pointer(node_list)))
    return (char_ps, node_list, node, cast(pointer(node), c_void_p))

def _event_generator(handle):
    if False:
        for i in range(10):
            print('nop')
    while True:
        event = _mpv_wait_event(handle, -1).contents
        if event.event_id.value == MpvEventID.NONE:
            raise StopIteration()
        yield event
_py_to_mpv = lambda name: name.replace('_', '-')
_mpv_to_py = lambda name: name.replace('-', '_')
_drop_nones = lambda *args: [arg for arg in args if arg is not None]

class _Proxy:

    def __init__(self, mpv):
        if False:
            for i in range(10):
                print('nop')
        super().__setattr__('mpv', mpv)

class _PropertyProxy(_Proxy):

    def __dir__(self):
        if False:
            for i in range(10):
                print('nop')
        return super().__dir__() + [name.replace('-', '_') for name in self.mpv.property_list]

class _FileLocalProxy(_Proxy):

    def __getitem__(self, name):
        if False:
            while True:
                i = 10
        return self.mpv.__getitem__(name, file_local=True)

    def __setitem__(self, name, value):
        if False:
            i = 10
            return i + 15
        return self.mpv.__setitem__(name, value, file_local=True)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.mpv)

class _OSDPropertyProxy(_PropertyProxy):

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        return self.mpv._get_property(_py_to_mpv(name), fmt=MpvFormat.OSD_STRING)

    def __setattr__(self, _name, _value):
        if False:
            for i in range(10):
                print('nop')
        raise AttributeError('OSD properties are read-only. Please use the regular property API for writing.')

class _DecoderPropertyProxy(_PropertyProxy):

    def __init__(self, mpv, decoder):
        if False:
            return 10
        super().__init__(mpv)
        super().__setattr__('_decoder', decoder)

    def __getattr__(self, name):
        if False:
            return 10
        return self.mpv._get_property(_py_to_mpv(name), decoder=self._decoder)

    def __setattr__(self, name, value):
        if False:
            return 10
        setattr(self.mpv, _py_to_mpv(name), value)

class GeneratorStream:
    """Transform a python generator into an mpv-compatible stream object. This only supports size() and read(), and
    does not support seek(), close() or cancel().
    """

    def __init__(self, generator_fun, size=None):
        if False:
            for i in range(10):
                print('nop')
        self._generator_fun = generator_fun
        self.size = size

    def seek(self, offset):
        if False:
            while True:
                i = 10
        self._read_iter = iter(self._generator_fun())
        self._read_chunk = b''
        return 0

    def read(self, size):
        if False:
            return 10
        if not self._read_chunk:
            try:
                self._read_chunk += next(self._read_iter)
            except StopIteration:
                return b''
        (rv, self._read_chunk) = (self._read_chunk[:size], self._read_chunk[size:])
        return rv

    def close(self):
        if False:
            i = 10
            return i + 15
        self._read_iter = iter([])

    def cancel(self):
        if False:
            while True:
                i = 10
        self._read_iter = iter([])

class ImageOverlay:

    def __init__(self, m, overlay_id, img=None, pos=(0, 0)):
        if False:
            return 10
        self.m = m
        self.overlay_id = overlay_id
        self.pos = pos
        self._size = None
        if img is not None:
            self.update(img)

    def update(self, img=None, pos=None):
        if False:
            i = 10
            return i + 15
        from PIL import Image
        if img is not None:
            self.img = img
        img = self.img
        (w, h) = img.size
        stride = w * 4
        if pos is not None:
            self.pos = pos
        (x, y) = self.pos
        bg = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        out = Image.alpha_composite(bg, img)
        if img.size != self._size:
            self._buf = create_string_buffer(w * h * 4)
            self._size = img.size
        self._buf[:] = out.tobytes('raw', 'BGRA')
        source = '&' + str(addressof(self._buf))
        self.m.overlay_add(self.overlay_id, x, y, source, 0, 'bgra', w, h, stride)

    def remove(self):
        if False:
            print('Hello World!')
        self.m.remove_overlay(self.overlay_id)

class FileOverlay:

    def __init__(self, m, overlay_id, filename=None, size=None, stride=None, pos=(0, 0)):
        if False:
            print('Hello World!')
        self.m = m
        self.overlay_id = overlay_id
        self.pos = pos
        self.size = size
        self.stride = stride
        if filename is not None:
            self.update(filename)

    def update(self, filename=None, size=None, stride=None, pos=None):
        if False:
            for i in range(10):
                print('nop')
        if filename is not None:
            self.filename = filename
        if pos is not None:
            self.pos = pos
        if size is not None:
            self.size = size
        if stride is not None:
            self.stride = stride
        (x, y) = self.pos
        (w, h) = self.size
        stride = self.stride or 4 * w
        self.m.overlay_add(self, self.overlay_id, x, y, self.filename, 0, 'bgra', w, h, stride)

    def remove(self):
        if False:
            while True:
                i = 10
        self.m.remove_overlay(self.overlay_id)

class MPV(object):
    """See man mpv(1) for the details of the implemented commands. All mpv properties can be accessed as
    ``my_mpv.some_property`` and all mpv options can be accessed as ``my_mpv['some-option']``.
    By default, properties are returned as decoded ``str`` and an error is thrown if the value does not contain valid
    utf-8. To get a decoded ``str`` if possibly but ``bytes`` instead of an error if not, use
    ``my_mpv.lazy.some_property``. To always get raw ``bytes``, use ``my_mpv.raw.some_property``.  To access a
    property's decoded OSD value, use ``my_mpv.osd.some_property``.
    To get API information on an option, use ``my_mpv.option_info('option-name')``. To get API information on a
    property, use ``my_mpv.properties['property-name']``. Take care to use mpv's dashed-names instead of the
    underscore_names exposed on the python object.
    To make your program not barf hard the first time its used on a weird file system **always** access properties
    containing file names or file tags through ``MPV.raw``.  """

    def __init__(self, *extra_mpv_flags, log_handler=None, start_event_thread=True, loglevel=None, **extra_mpv_opts):
        if False:
            while True:
                i = 10
        'Create an MPV instance.\n        Extra arguments and extra keyword arguments will be passed to mpv as options.\n        '
        self.handle = _mpv_create()
        self._event_thread = None
        self._core_shutdown = False
        _mpv_set_option_string(self.handle, b'audio-display', b'no')
        istr = lambda o: ('yes' if o else 'no') if type(o) is bool else str(o)
        try:
            for flag in extra_mpv_flags:
                _mpv_set_option_string(self.handle, flag.encode('utf-8'), b'')
            for (k, v) in extra_mpv_opts.items():
                _mpv_set_option_string(self.handle, k.replace('_', '-').encode('utf-8'), istr(v).encode('utf-8'))
        finally:
            _mpv_initialize(self.handle)
        self.osd = _OSDPropertyProxy(self)
        self.file_local = _FileLocalProxy(self)
        self.raw = _DecoderPropertyProxy(self, identity_decoder)
        self.strict = _DecoderPropertyProxy(self, strict_decoder)
        self.lazy = _DecoderPropertyProxy(self, lazy_decoder)
        self._event_callbacks = []
        self._event_handler_lock = threading.Lock()
        self._property_handlers = collections.defaultdict(lambda : [])
        self._quit_handlers = set()
        self._message_handlers = {}
        self._key_binding_handlers = {}
        self._event_handle = _mpv_create_client(self.handle, b'py_event_handler')
        self._log_handler = log_handler
        self._stream_protocol_cbs = {}
        self._stream_protocol_frontends = collections.defaultdict(lambda : {})
        self.register_stream_protocol('python', self._python_stream_open)
        self._python_streams = {}
        self._python_stream_catchall = None
        self.overlay_ids = set()
        self.overlays = {}
        if loglevel is not None or log_handler is not None:
            self.set_loglevel(loglevel or 'terminal-default')
        if start_event_thread:
            self._event_thread = threading.Thread(target=self._loop, name='MPVEventHandlerThread')
            self._event_thread.setDaemon(True)
            self._event_thread.start()
        else:
            self._event_thread = None

    def _loop(self):
        if False:
            i = 10
            return i + 15
        for event in _event_generator(self._event_handle):
            try:
                devent = event.as_dict(decoder=lazy_decoder)
                eid = devent['event_id']
                with self._event_handler_lock:
                    if eid == MpvEventID.SHUTDOWN:
                        self._core_shutdown = True
                for callback in self._event_callbacks:
                    callback(devent)
                if eid == MpvEventID.PROPERTY_CHANGE:
                    pc = devent['event']
                    (name, value, _fmt) = (pc['name'], pc['value'], pc['format'])
                    for handler in self._property_handlers[name]:
                        handler(name, value)
                if eid == MpvEventID.LOG_MESSAGE and self._log_handler is not None:
                    ev = devent['event']
                    self._log_handler(ev['level'], ev['prefix'], ev['text'])
                if eid == MpvEventID.CLIENT_MESSAGE:
                    (target, *args) = devent['event']['args']
                    if target in self._message_handlers:
                        self._message_handlers[target](*args)
                if eid == MpvEventID.SHUTDOWN:
                    _mpv_destroy(self._event_handle)
                    return
            except Exception as e:
                print('Exception inside python-mpv event loop:', file=sys.stderr)
                traceback.print_exc()

    @property
    def core_shutdown(self):
        if False:
            while True:
                i = 10
        'Property indicating whether the core has been shut down. Possible causes for this are e.g. the `quit` command\n        or a user closing the mpv window.'
        return self._core_shutdown

    def check_core_alive(self):
        if False:
            i = 10
            return i + 15
        ' This method can be used as a sanity check to tests whether the core is still alive at the time it is\n        called.'
        if self._core_shutdown:
            raise ShutdownError('libmpv core has been shutdown')

    def wait_until_paused(self):
        if False:
            while True:
                i = 10
        'Waits until playback of the current title is paused or done. Raises a ShutdownError if the core is shutdown while\n        waiting.'
        self.wait_for_property('core-idle')

    def wait_for_playback(self):
        if False:
            return 10
        'Waits until playback of the current title is finished. Raises a ShutdownError if the core is shutdown while\n        waiting.\n        '
        self.wait_for_event('end_file')

    def wait_until_playing(self):
        if False:
            i = 10
            return i + 15
        'Waits until playback of the current title has started. Raises a ShutdownError if the core is shutdown while\n        waiting.'
        self.wait_for_property('core-idle', lambda idle: not idle)

    def wait_for_property(self, name, cond=lambda val: val, level_sensitive=True):
        if False:
            i = 10
            return i + 15
        'Waits until ``cond`` evaluates to a truthy value on the named property. This can be used to wait for\n        properties such as ``idle_active`` indicating the player is done with regular playback and just idling around.\n        Raises a ShutdownError when the core is shutdown while waiting.\n        '
        with self.prepare_and_wait_for_property(name, cond, level_sensitive):
            pass

    def wait_for_shutdown(self):
        if False:
            i = 10
            return i + 15
        'Wait for core to shutdown (e.g. through quit() or terminate()).'
        sema = threading.Semaphore(value=0)

        @self.event_callback('shutdown')
        def shutdown_handler(event):
            if False:
                return 10
            sema.release()
        sema.acquire()
        shutdown_handler.unregister_mpv_events()

    @contextmanager
    def prepare_and_wait_for_property(self, name, cond=lambda val: val, level_sensitive=True):
        if False:
            while True:
                i = 10
        'Context manager that waits until ``cond`` evaluates to a truthy value on the named property. See\n        prepare_and_wait_for_event for usage.\n        Raises a ShutdownError when the core is shutdown while waiting.\n        '
        sema = threading.Semaphore(value=0)

        def observer(name, val):
            if False:
                for i in range(10):
                    print('nop')
            if cond(val):
                sema.release()
        self.observe_property(name, observer)

        @self.event_callback('shutdown')
        def shutdown_handler(event):
            if False:
                print('Hello World!')
            sema.release()
        yield
        if not level_sensitive or not cond(getattr(self, name.replace('-', '_'))):
            sema.acquire()
        self.check_core_alive()
        shutdown_handler.unregister_mpv_events()
        self.unobserve_property(name, observer)

    def wait_for_event(self, *event_types, cond=lambda evt: True):
        if False:
            print('Hello World!')
        "Waits for the indicated event(s). If cond is given, waits until cond(event) is true. Raises a ShutdownError\n        if the core is shutdown while waiting. This also happens when 'shutdown' is in event_types.\n        "
        with self.prepare_and_wait_for_event(*event_types, cond=cond):
            pass

    @contextmanager
    def prepare_and_wait_for_event(self, *event_types, cond=lambda evt: True):
        if False:
            while True:
                i = 10
        "Context manager that waits for the indicated event(s) like wait_for_event after running. If cond is given,\n        waits until cond(event) is true. Raises a ShutdownError if the core is shutdown while waiting. This also happens\n        when 'shutdown' is in event_types.\n        Compared to wait_for_event this handles the case where a thread waits for an event it itself causes in a\n        thread-safe way. An example from the testsuite is:\n        with self.m.prepare_and_wait_for_event('client_message'):\n            self.m.keypress(key)\n        Using just wait_for_event it would be impossible to ensure the event is caught since it may already have been\n        handled in the interval between keypress(...) running and a subsequent wait_for_event(...) call.\n        "
        sema = threading.Semaphore(value=0)

        @self.event_callback('shutdown')
        def shutdown_handler(event):
            if False:
                while True:
                    i = 10
            sema.release()

        @self.event_callback(*event_types)
        def target_handler(evt):
            if False:
                print('Hello World!')
            if cond(evt):
                sema.release()
        yield
        sema.acquire()
        self.check_core_alive()
        shutdown_handler.unregister_mpv_events()
        target_handler.unregister_mpv_events()

    def __del__(self):
        if False:
            i = 10
            return i + 15
        if self.handle:
            self.terminate()

    def terminate(self):
        if False:
            return 10
        "Properly terminates this player instance. Preferably use this instead of relying on python's garbage\n        collector to cause this to be called from the object's destructor.\n        This method will detach the main libmpv handle and wait for mpv to shut down and the event thread to finish.\n        "
        (self.handle, handle) = (None, self.handle)
        if threading.current_thread() is self._event_thread:
            raise UserWarning('terminate() should not be called from event thread (e.g. from a callback function). If you want to terminate mpv from here, please call quit() instead, then sync the main thread against the event thread using e.g. wait_for_shutdown(), then terminate() from the main thread. This call has been transformed into a call to quit().')
            self.quit()
        else:
            _mpv_terminate_destroy(handle)
            if self._event_thread:
                self._event_thread.join()

    def set_loglevel(self, level):
        if False:
            while True:
                i = 10
        'Set MPV\'s log level. This adjusts which output will be sent to this object\'s log handlers. If you just want\n        mpv\'s regular terminal output, you don\'t need to adjust this but just need to pass a log handler to the MPV\n        constructur such as ``MPV(log_handler=print)``.\n        Valid log levels are "no", "fatal", "error", "warn", "info", "v" "debug" and "trace". For details see your mpv\'s\n        client.h header file.\n        '
        _mpv_request_log_messages(self._event_handle, level.encode('utf-8'))

    def command(self, name, *args):
        if False:
            print('Hello World!')
        'Execute a raw command.'
        args = [name.encode('utf-8')] + [arg if type(arg) is bytes else str(arg).encode('utf-8') for arg in args if arg is not None] + [None]
        _mpv_command(self.handle, (c_char_p * len(args))(*args))

    def node_command(self, name, *args, decoder=strict_decoder):
        if False:
            print('Hello World!')
        (_1, _2, _3, pointer) = _make_node_str_list([name, *args])
        out = cast(create_string_buffer(sizeof(MpvNode)), POINTER(MpvNode))
        ppointer = cast(pointer, POINTER(MpvNode))
        _mpv_command_node(self.handle, ppointer, out)
        rv = out.contents.node_value(decoder=decoder)
        _mpv_free_node_contents(out)
        return rv

    def seek(self, amount, reference='relative', precision='default-precise'):
        if False:
            while True:
                i = 10
        'Mapped mpv seek command, see man mpv(1).'
        self.command('seek', amount, reference, precision)

    def revert_seek(self):
        if False:
            for i in range(10):
                print('nop')
        'Mapped mpv revert_seek command, see man mpv(1).'
        self.command('revert_seek')

    def frame_step(self):
        if False:
            return 10
        'Mapped mpv frame-step command, see man mpv(1).'
        self.command('frame-step')

    def frame_back_step(self):
        if False:
            i = 10
            return i + 15
        'Mapped mpv frame_back_step command, see man mpv(1).'
        self.command('frame_back_step')

    def property_add(self, name, value=1):
        if False:
            while True:
                i = 10
        "Add the given value to the property's value. On overflow or underflow, clamp the property to the maximum. If\n        ``value`` is omitted, assume ``1``.\n        "
        self.command('add', name, value)

    def property_multiply(self, name, factor):
        if False:
            i = 10
            return i + 15
        'Multiply the value of a property with a numeric factor.'
        self.command('multiply', name, factor)

    def cycle(self, name, direction='up'):
        if False:
            while True:
                i = 10
        'Cycle the given property. ``up`` and ``down`` set the cycle direction. On overflow, set the property back to\n        the minimum, on underflow set it to the maximum. If ``up`` or ``down`` is omitted, assume ``up``.\n        '
        self.command('cycle', name, direction)

    def screenshot(self, includes='subtitles', mode='single'):
        if False:
            print('Hello World!')
        'Mapped mpv screenshot command, see man mpv(1).'
        self.command('screenshot', includes, mode)

    def screenshot_to_file(self, filename, includes='subtitles'):
        if False:
            i = 10
            return i + 15
        'Mapped mpv screenshot_to_file command, see man mpv(1).'
        self.command('screenshot_to_file', filename.encode(fs_enc), includes)

    def screenshot_raw(self, includes='subtitles'):
        if False:
            return 10
        'Mapped mpv screenshot_raw command, see man mpv(1). Returns a pillow Image object.'
        from PIL import Image
        res = self.node_command('screenshot-raw', includes)
        if res['format'] != 'bgr0':
            raise ValueError('Screenshot in unknown format "{}". Currently, only bgr0 is supported.'.format(res['format']))
        img = Image.frombytes('RGBA', (res['stride'] // 4, res['h']), res['data'])
        (b, g, r, a) = img.split()
        return Image.merge('RGB', (r, g, b))

    def allocate_overlay_id(self):
        if False:
            print('Hello World!')
        free_ids = set(range(64)) - self.overlay_ids
        if not free_ids:
            raise IndexError('All overlay IDs are in use')
        (next_id, *_) = sorted(free_ids)
        self.overlay_ids.add(next_id)
        return next_id

    def free_overlay_id(self, overlay_id):
        if False:
            print('Hello World!')
        self.overlay_ids.remove(overlay_id)

    def create_file_overlay(self, filename=None, size=None, stride=None, pos=(0, 0)):
        if False:
            return 10
        overlay_id = self.allocate_overlay_id()
        overlay = FileOverlay(self, overlay_id, filename, size, stride, pos)
        self.overlays[overlay_id] = overlay
        return overlay

    def create_image_overlay(self, img=None, pos=(0, 0)):
        if False:
            return 10
        overlay_id = self.allocate_overlay_id()
        overlay = ImageOverlay(self, overlay_id, img, pos)
        self.overlays[overlay_id] = overlay
        return overlay

    def remove_overlay(self, overlay_id):
        if False:
            return 10
        self.overlay_remove(overlay_id)
        self.free_overlay_id(overlay_id)
        del self.overlays[overlay_id]

    def playlist_next(self, mode='weak'):
        if False:
            for i in range(10):
                print('nop')
        'Mapped mpv playlist_next command, see man mpv(1).'
        self.command('playlist_next', mode)

    def playlist_prev(self, mode='weak'):
        if False:
            while True:
                i = 10
        'Mapped mpv playlist_prev command, see man mpv(1).'
        self.command('playlist_prev', mode)

    def playlist_play_index(self, idx):
        if False:
            for i in range(10):
                print('nop')
        'Mapped mpv playlist-play-index command, see man mpv(1).'
        self.command('playlist-play-index', idx)

    @staticmethod
    def _encode_options(options):
        if False:
            i = 10
            return i + 15
        return ','.join(('{}={}'.format(_py_to_mpv(str(key)), str(val)) for (key, val) in options.items()))

    def loadfile(self, filename, mode='replace', **options):
        if False:
            i = 10
            return i + 15
        'Mapped mpv loadfile command, see man mpv(1).'
        self.command('loadfile', filename.encode(fs_enc), mode, MPV._encode_options(options))

    def loadlist(self, playlist, mode='replace'):
        if False:
            for i in range(10):
                print('nop')
        'Mapped mpv loadlist command, see man mpv(1).'
        self.command('loadlist', playlist.encode(fs_enc), mode)

    def playlist_clear(self):
        if False:
            i = 10
            return i + 15
        'Mapped mpv playlist_clear command, see man mpv(1).'
        self.command('playlist_clear')

    def playlist_remove(self, index='current'):
        if False:
            while True:
                i = 10
        'Mapped mpv playlist_remove command, see man mpv(1).'
        self.command('playlist_remove', index)

    def playlist_move(self, index1, index2):
        if False:
            for i in range(10):
                print('nop')
        'Mapped mpv playlist_move command, see man mpv(1).'
        self.command('playlist_move', index1, index2)

    def playlist_shuffle(self):
        if False:
            return 10
        'Mapped mpv playlist-shuffle command, see man mpv(1).'
        self.command('playlist-shuffle')

    def playlist_unshuffle(self):
        if False:
            print('Hello World!')
        'Mapped mpv playlist-unshuffle command, see man mpv(1).'
        self.command('playlist-unshuffle')

    def run(self, command, *args):
        if False:
            print('Hello World!')
        'Mapped mpv run command, see man mpv(1).'
        self.command('run', command, *args)

    def quit(self, code=None):
        if False:
            while True:
                i = 10
        'Mapped mpv quit command, see man mpv(1).'
        self.command('quit', code)

    def quit_watch_later(self, code=None):
        if False:
            return 10
        'Mapped mpv quit_watch_later command, see man mpv(1).'
        self.command('quit_watch_later', code)

    def stop(self, keep_playlist=False):
        if False:
            print('Hello World!')
        'Mapped mpv stop command, see man mpv(1).'
        if keep_playlist:
            self.command('stop', 'keep-playlist')
        else:
            self.command('stop')

    def audio_add(self, url, flags='select', title=None, lang=None):
        if False:
            print('Hello World!')
        'Mapped mpv audio_add command, see man mpv(1).'
        self.command('audio_add', url.encode(fs_enc), *_drop_nones(flags, title, lang))

    def audio_remove(self, audio_id=None):
        if False:
            i = 10
            return i + 15
        'Mapped mpv audio_remove command, see man mpv(1).'
        self.command('audio_remove', audio_id)

    def audio_reload(self, audio_id=None):
        if False:
            print('Hello World!')
        'Mapped mpv audio_reload command, see man mpv(1).'
        self.command('audio_reload', audio_id)

    def video_add(self, url, flags='select', title=None, lang=None):
        if False:
            i = 10
            return i + 15
        'Mapped mpv video_add command, see man mpv(1).'
        self.command('video_add', url.encode(fs_enc), *_drop_nones(flags, title, lang))

    def video_remove(self, video_id=None):
        if False:
            for i in range(10):
                print('nop')
        'Mapped mpv video_remove command, see man mpv(1).'
        self.command('video_remove', video_id)

    def video_reload(self, video_id=None):
        if False:
            i = 10
            return i + 15
        'Mapped mpv video_reload command, see man mpv(1).'
        self.command('video_reload', video_id)

    def sub_add(self, url, flags='select', title=None, lang=None):
        if False:
            i = 10
            return i + 15
        'Mapped mpv sub_add command, see man mpv(1).'
        self.command('sub_add', url.encode(fs_enc), *_drop_nones(flags, title, lang))

    def sub_remove(self, sub_id=None):
        if False:
            for i in range(10):
                print('nop')
        'Mapped mpv sub_remove command, see man mpv(1).'
        self.command('sub_remove', sub_id)

    def sub_reload(self, sub_id=None):
        if False:
            return 10
        'Mapped mpv sub_reload command, see man mpv(1).'
        self.command('sub_reload', sub_id)

    def sub_step(self, skip):
        if False:
            print('Hello World!')
        'Mapped mpv sub_step command, see man mpv(1).'
        self.command('sub_step', skip)

    def sub_seek(self, skip):
        if False:
            return 10
        'Mapped mpv sub_seek command, see man mpv(1).'
        self.command('sub_seek', skip)

    def toggle_osd(self):
        if False:
            while True:
                i = 10
        'Mapped mpv osd command, see man mpv(1).'
        self.command('osd')

    def print_text(self, text):
        if False:
            print('Hello World!')
        'Mapped mpv print-text command, see man mpv(1).'
        self.command('print-text', text)

    def show_text(self, string, duration='-1', level=None):
        if False:
            for i in range(10):
                print('nop')
        'Mapped mpv show_text command, see man mpv(1).'
        self.command('show_text', string, duration, level)

    def expand_text(self, text):
        if False:
            while True:
                i = 10
        'Mapped mpv expand-text command, see man mpv(1).'
        return self.node_command('expand-text', text)

    def expand_path(self, path):
        if False:
            i = 10
            return i + 15
        'Mapped mpv expand-path command, see man mpv(1).'
        return self.node_command('expand-path', path)

    def show_progress(self):
        if False:
            print('Hello World!')
        'Mapped mpv show_progress command, see man mpv(1).'
        self.command('show_progress')

    def rescan_external_files(self, mode='reselect'):
        if False:
            print('Hello World!')
        'Mapped mpv rescan-external-files command, see man mpv(1).'
        self.command('rescan-external-files', mode)

    def discnav(self, command):
        if False:
            while True:
                i = 10
        'Mapped mpv discnav command, see man mpv(1).'
        self.command('discnav', command)

    def mouse(x, y, button=None, mode='single'):
        if False:
            print('Hello World!')
        'Mapped mpv mouse command, see man mpv(1).'
        if button is None:
            self.command('mouse', x, y, mode)
        else:
            self.command('mouse', x, y, button, mode)

    def keypress(self, name):
        if False:
            while True:
                i = 10
        'Mapped mpv keypress command, see man mpv(1).'
        self.command('keypress', name)

    def keydown(self, name):
        if False:
            return 10
        'Mapped mpv keydown command, see man mpv(1).'
        self.command('keydown', name)

    def keyup(self, name=None):
        if False:
            i = 10
            return i + 15
        'Mapped mpv keyup command, see man mpv(1).'
        if name is None:
            self.command('keyup')
        else:
            self.command('keyup', name)

    def keybind(self, name, command):
        if False:
            for i in range(10):
                print('nop')
        'Mapped mpv keybind command, see man mpv(1).'
        self.command('keybind', name, command)

    def write_watch_later_config(self):
        if False:
            while True:
                i = 10
        'Mapped mpv write_watch_later_config command, see man mpv(1).'
        self.command('write_watch_later_config')

    def overlay_add(self, overlay_id, x, y, file_or_fd, offset, fmt, w, h, stride):
        if False:
            while True:
                i = 10
        'Mapped mpv overlay_add command, see man mpv(1).'
        self.command('overlay_add', overlay_id, x, y, file_or_fd, offset, fmt, w, h, stride)

    def overlay_remove(self, overlay_id):
        if False:
            print('Hello World!')
        'Mapped mpv overlay_remove command, see man mpv(1).'
        self.command('overlay_remove', overlay_id)

    def script_message(self, *args):
        if False:
            for i in range(10):
                print('nop')
        'Mapped mpv script_message command, see man mpv(1).'
        self.command('script_message', *args)

    def script_message_to(self, target, *args):
        if False:
            i = 10
            return i + 15
        'Mapped mpv script_message_to command, see man mpv(1).'
        self.command('script_message_to', target, *args)

    def observe_property(self, name, handler):
        if False:
            i = 10
            return i + 15
        'Register an observer on the named property. An observer is a function that is called with the new property\n        value every time the property\'s value is changed. The basic function signature is ``fun(property_name,\n        new_value)`` with new_value being the decoded property value as a python object. This function can be used as a\n        function decorator if no handler is given.\n        To unregister the observer, call either of ``mpv.unobserve_property(name, handler)``,\n        ``mpv.unobserve_all_properties(handler)`` or the handler\'s ``unregister_mpv_properties`` attribute::\n            @player.observe_property(\'volume\')\n            def my_handler(new_volume, *):\n                print("It\'s loud!", volume)\n            my_handler.unregister_mpv_properties()\n        exit_handler is a function taking no arguments that is called when the underlying mpv handle is terminated (e.g.\n        from calling MPV.terminate() or issuing a "quit" input command).\n        '
        self._property_handlers[name].append(handler)
        _mpv_observe_property(self._event_handle, hash(name) & 18446744073709551615, name.encode('utf-8'), MpvFormat.NODE)

    def property_observer(self, name):
        if False:
            i = 10
            return i + 15
        'Function decorator to register a property observer. See ``MPV.observe_property`` for details.'

        def wrapper(fun):
            if False:
                while True:
                    i = 10
            self.observe_property(name, fun)
            fun.unobserve_mpv_properties = lambda : self.unobserve_property(name, fun)
            return fun
        return wrapper

    def unobserve_property(self, name, handler):
        if False:
            i = 10
            return i + 15
        "Unregister a property observer. This requires both the observed property's name and the handler function that\n        was originally registered as one handler could be registered for several properties. To unregister a handler\n        from *all* observed properties see ``unobserve_all_properties``.\n        "
        self._property_handlers[name].remove(handler)
        if not self._property_handlers[name]:
            _mpv_unobserve_property(self._event_handle, hash(name) & 18446744073709551615)

    def unobserve_all_properties(self, handler):
        if False:
            i = 10
            return i + 15
        'Unregister a property observer from *all* observed properties.'
        for name in self._property_handlers:
            self.unobserve_property(name, handler)

    def register_message_handler(self, target, handler=None):
        if False:
            return 10
        "Register a mpv script message handler. This can be used to communicate with embedded lua scripts. Pass the\n        script message target name this handler should be listening to and the handler function.\n        WARNING: Only one handler can be registered at a time for any given target.\n        To unregister the message handler, call its ``unregister_mpv_messages`` function::\n            player = mpv.MPV()\n            @player.message_handler('foo')\n            def my_handler(some, args):\n                print(args)\n            my_handler.unregister_mpv_messages()\n        "
        self._register_message_handler_internal(target, handler)

    def _register_message_handler_internal(self, target, handler):
        if False:
            i = 10
            return i + 15
        self._message_handlers[target] = handler

    def unregister_message_handler(self, target_or_handler):
        if False:
            return 10
        'Unregister a mpv script message handler for the given script message target name.\n        You can also call the ``unregister_mpv_messages`` function attribute set on the handler function when it is\n        registered.\n        '
        if isinstance(target_or_handler, str):
            del self._message_handlers[target_or_handler]
        else:
            for (key, val) in self._message_handlers.items():
                if val == target_or_handler:
                    del self._message_handlers[key]

    def message_handler(self, target):
        if False:
            return 10
        "Decorator to register a mpv script message handler.\n        WARNING: Only one handler can be registered at a time for any given target.\n        To unregister the message handler, call its ``unregister_mpv_messages`` function::\n            player = mpv.MPV()\n            @player.message_handler('foo')\n            def my_handler(some, args):\n                print(args)\n            my_handler.unregister_mpv_messages()\n        "

        def register(handler):
            if False:
                while True:
                    i = 10
            self._register_message_handler_internal(target, handler)
            handler.unregister_mpv_messages = lambda : self.unregister_message_handler(handler)
            return handler
        return register

    def register_event_callback(self, callback):
        if False:
            while True:
                i = 10
        "Register a blanket event callback receiving all event types.\n        To unregister the event callback, call its ``unregister_mpv_events`` function::\n            player = mpv.MPV()\n            @player.event_callback('shutdown')\n            def my_handler(event):\n                print('It ded.')\n            my_handler.unregister_mpv_events()\n        "
        self._event_callbacks.append(callback)

    def unregister_event_callback(self, callback):
        if False:
            print('Hello World!')
        'Unregiser an event callback.'
        self._event_callbacks.remove(callback)

    def event_callback(self, *event_types):
        if False:
            for i in range(10):
                print('nop')
        "Function decorator to register a blanket event callback for the given event types. Event types can be given\n        as str (e.g.  'start-file'), integer or MpvEventID object.\n        WARNING: Due to the way this is filtering events, this decorator cannot be chained with itself.\n        To unregister the event callback, call its ``unregister_mpv_events`` function::\n            player = mpv.MPV()\n            @player.event_callback('shutdown')\n            def my_handler(event):\n                print('It ded.')\n            my_handler.unregister_mpv_events()\n        "

        def register(callback):
            if False:
                for i in range(10):
                    print('nop')
            with self._event_handler_lock:
                self.check_core_alive()
                types = [MpvEventID.from_str(t) if isinstance(t, str) else t for t in event_types] or MpvEventID.ANY

                @wraps(callback)
                def wrapper(event, *args, **kwargs):
                    if False:
                        while True:
                            i = 10
                    if event['event_id'] in types:
                        callback(event, *args, **kwargs)
                self._event_callbacks.append(wrapper)
                wrapper.unregister_mpv_events = partial(self.unregister_event_callback, wrapper)
                return wrapper
        return register

    @staticmethod
    def _binding_name(callback_or_cmd):
        if False:
            for i in range(10):
                print('nop')
        return 'py_kb_{:016x}'.format(hash(callback_or_cmd) & 18446744073709551615)

    def on_key_press(self, keydef, mode='force'):
        if False:
            for i in range(10):
                print('nop')
        "Function decorator to register a simplified key binding. The callback is called whenever the key given is\n        *pressed*.\n        To unregister the callback function, you can call its ``unregister_mpv_key_bindings`` attribute::\n            player = mpv.MPV()\n            @player.on_key_press('Q')\n            def binding():\n                print('blep')\n            binding.unregister_mpv_key_bindings()\n        WARNING: For a single keydef only a single callback/command can be registered at the same time. If you register\n        a binding multiple times older bindings will be overwritten and there is a possibility of references leaking. So\n        don't do that.\n        The BIG FAT WARNING regarding untrusted keydefs from the key_binding method applies here as well.\n        "

        def register(fun):
            if False:
                print('Hello World!')

            @self.key_binding(keydef, mode)
            @wraps(fun)
            def wrapper(state='p-', name=None, char=None):
                if False:
                    for i in range(10):
                        print('nop')
                if state[0] in ('d', 'p'):
                    fun()
            return wrapper
        return register

    def key_binding(self, keydef, mode='force'):
        if False:
            while True:
                i = 10
        'Function decorator to register a low-level key binding.\n        The callback function signature is ``fun(key_state, key_name)`` where ``key_state`` is either ``\'U\'`` for "key\n        up" or ``\'D\'`` for "key down".\n        The keydef format is: ``[Shift+][Ctrl+][Alt+][Meta+]<key>`` where ``<key>`` is either the literal character the\n        key produces (ASCII or Unicode character), or a symbolic name (as printed by ``mpv --input-keylist``).\n        To unregister the callback function, you can call its ``unregister_mpv_key_bindings`` attribute::\n            player = mpv.MPV()\n            @player.key_binding(\'Q\')\n            def binding(state, name, char):\n                print(\'blep\')\n            binding.unregister_mpv_key_bindings()\n        WARNING: For a single keydef only a single callback/command can be registered at the same time. If you register\n        a binding multiple times older bindings will be overwritten and there is a possibility of references leaking. So\n        don\'t do that.\n        BIG FAT WARNING: mpv\'s key binding mechanism is pretty powerful.  This means, you essentially get arbitrary code\n        exectution through key bindings. This interface makes some limited effort to sanitize the keydef given in the\n        first parameter, but YOU SHOULD NOT RELY ON THIS IN FOR SECURITY. If your input comes from config files, this is\n        completely fine--but, if you are about to pass untrusted input into this parameter, better double-check whether\n        this is secure in your case.\n        '

        def register(fun):
            if False:
                for i in range(10):
                    print('nop')
            fun.mpv_key_bindings = getattr(fun, 'mpv_key_bindings', []) + [keydef]

            def unregister_all():
                if False:
                    print('Hello World!')
                for keydef in fun.mpv_key_bindings:
                    self.unregister_key_binding(keydef)
            fun.unregister_mpv_key_bindings = unregister_all
            self.register_key_binding(keydef, fun, mode)
            return fun
        return register

    def register_key_binding(self, keydef, callback_or_cmd, mode='force'):
        if False:
            return 10
        'Register a key binding. This takes an mpv keydef and either a string containing a mpv command or a python\n        callback function.  See ``MPV.key_binding`` for details.\n        '
        if not re.match('(Shift+)?(Ctrl+)?(Alt+)?(Meta+)?(.|\\w+)', keydef):
            raise ValueError('Invalid keydef. Expected format: [Shift+][Ctrl+][Alt+][Meta+]<key>\n<key> is either the literal character the key produces (ASCII or Unicode character), or a symbolic name (as printed by --input-keylist')
        binding_name = MPV._binding_name(keydef)
        if callable(callback_or_cmd):
            self._key_binding_handlers[binding_name] = callback_or_cmd
            self.register_message_handler('key-binding', self._handle_key_binding_message)
            self.command('define-section', binding_name, '{} script-binding py_event_handler/{}'.format(keydef, binding_name), mode)
        elif isinstance(callback_or_cmd, str):
            self.command('define-section', binding_name, '{} {}'.format(keydef, callback_or_cmd), mode)
        else:
            raise TypeError('register_key_binding expects either an str with an mpv command or a python callable.')
        self.command('enable-section', binding_name, 'allow-hide-cursor+allow-vo-dragging')

    def _handle_key_binding_message(self, binding_name, key_state, key_name=None, key_char=None):
        if False:
            i = 10
            return i + 15
        self._key_binding_handlers[binding_name](key_state, key_name, key_char)

    def unregister_key_binding(self, keydef):
        if False:
            while True:
                i = 10
        'Unregister a key binding by keydef.'
        binding_name = MPV._binding_name(keydef)
        self.command('disable-section', binding_name)
        self.command('define-section', binding_name, '')
        if binding_name in self._key_binding_handlers:
            del self._key_binding_handlers[binding_name]
            if not self._key_binding_handlers:
                self.unregister_message_handler('key-binding')

    def register_stream_protocol(self, proto, open_fn=None):
        if False:
            i = 10
            return i + 15
        ' Register a custom stream protocol as documented in libmpv/stream_cb.h:\n            https://github.com/mpv-player/mpv/blob/master/libmpv/stream_cb.h\n            proto is the protocol scheme, e.g. "foo" for "foo://" urls.\n            This function can either be used with two parameters or it can be used as a decorator on the target\n            function.\n            open_fn is a function taking an URI string and returning an mpv stream object.\n            open_fn may raise a ValueError to signal libmpv the URI could not be opened.\n            The mpv stream protocol is as follows:\n            class Stream:\n                @property\n                def size(self):\n                    return None # unknown size\n                    return size # int with size in bytes\n                def read(self, size):\n                    ...\n                    return read # non-empty bytes object with input\n                    return b\'\' # empty byte object signals permanent EOF\n                def seek(self, pos):\n                    return new_offset # integer with new byte offset. The new offset may be before the requested offset\n                    in case an exact seek is inconvenient.\n                def close(self):\n                    ...\n                # def cancel(self): (future API versions only)\n                #     Abort a running read() or seek() operation\n                #     ...\n        '

        def decorator(open_fn):
            if False:
                for i in range(10):
                    print('nop')

            @StreamOpenFn
            def open_backend(_userdata, uri, cb_info):
                if False:
                    print('Hello World!')
                try:
                    frontend = open_fn(uri.decode('utf-8'))
                except ValueError:
                    return ErrorCode.LOADING_FAILED

                def read_backend(_userdata, buf, bufsize):
                    if False:
                        return 10
                    data = frontend.read(bufsize)
                    for i in range(len(data)):
                        buf[i] = data[i]
                    return len(data)
                cb_info.contents.cookie = None
                read = cb_info.contents.read = StreamReadFn(read_backend)
                close = cb_info.contents.close = StreamCloseFn(lambda _userdata: frontend.close())
                (seek, size, cancel) = (None, None, None)
                if hasattr(frontend, 'seek'):
                    seek = cb_info.contents.seek = StreamSeekFn(lambda _userdata, offx: frontend.seek(offx))
                if hasattr(frontend, 'size') and frontend.size is not None:
                    size = cb_info.contents.size = StreamSizeFn(lambda _userdata: frontend.size)
                frontend._registered_callbacks = [read, close, seek, size, cancel]
                self._stream_protocol_frontends[proto][uri] = frontend
                return 0
            if proto in self._stream_protocol_cbs:
                raise KeyError('Stream protocol already registered')
            self._stream_protocol_cbs[proto] = [open_backend]
            _mpv_stream_cb_add_ro(self.handle, proto.encode('utf-8'), c_void_p(), open_backend)
            return open_fn
        if open_fn is not None:
            decorator(open_fn)
        return decorator

    def play(self, filename):
        if False:
            print('Hello World!')
        'Play a path or URL (requires ``ytdl`` option to be set).'
        self.loadfile(filename)

    @property
    def playlist_filenames(self):
        if False:
            while True:
                i = 10
        'Return all playlist item file names/URLs as a list of strs.'
        return [element['filename'] for element in self.playlist]

    def playlist_append(self, filename, **options):
        if False:
            while True:
                i = 10
        "Append a path or URL to the playlist. This does not start playing the file automatically. To do that, use\n        ``MPV.loadfile(filename, 'append-play')``."
        self.loadfile(filename, 'append', **options)

    def _python_stream_open(self, uri):
        if False:
            for i in range(10):
                print('nop')
        'Internal handler for python:// protocol streams registered through @python_stream(...) and\n        @python_stream_catchall\n        '
        (name,) = re.fullmatch('python://(.*)', uri).groups()
        if name in self._python_streams:
            (generator_fun, size) = self._python_streams[name]
        elif self._python_stream_catchall is not None:
            (generator_fun, size) = self._python_stream_catchall(name)
        else:
            raise ValueError('Python stream name not found and no catch-all defined')
        return GeneratorStream(generator_fun, size)

    def python_stream(self, name=None, size=None):
        if False:
            return 10
        'Register a generator for the python stream with the given name.\n        name is the name, i.e. the part after the "python://" in the URI, that this generator is registered as.\n        size is the total number of bytes in the stream (if known).\n        Any given name can only be registered once. The catch-all can also only be registered once. To unregister a\n        stream, call the .unregister function set on the callback.\n        The generator signals EOF by returning, manually raising StopIteration or by yielding b\'\', an empty bytes\n        object.\n        The generator may be called multiple times if libmpv seeks or loops.\n        See also: @mpv.python_stream_catchall\n        @mpv.python_stream(\'foobar\')\n        def reader():\n            for chunk in chunks:\n                yield chunk\n        mpv.play(\'python://foobar\')\n        mpv.wait_for_playback()\n        reader.unregister()\n        '

        def register(cb):
            if False:
                print('Hello World!')
            if name in self._python_streams:
                raise KeyError('Python stream name "{}" is already registered'.format(name))
            self._python_streams[name] = (cb, size)

            def unregister():
                if False:
                    for i in range(10):
                        print('nop')
                if name not in self._python_streams or self._python_streams[name][0] is not cb:
                    raise RuntimeError('Python stream has already been unregistered')
                del self._python_streams[name]
            cb.unregister = unregister
            return cb
        return register

    def python_stream_catchall(self, cb):
        if False:
            while True:
                i = 10
        " Register a catch-all python stream to be called when no name matches can be found. Use this decorator on a\n        function that takes a name argument and returns a (generator, size) tuple (with size being None if unknown).\n        An invalid URI can be signalled to libmpv by raising a ValueError inside the callback.\n        See also: @mpv.python_stream(name, size)\n        @mpv.python_stream_catchall\n        def catchall(name):\n            if not name.startswith('foo'):\n                raise ValueError('Unknown Name')\n            def foo_reader():\n                with open(name, 'rb') as f:\n                    while True:\n                        chunk = f.read(1024)\n                        if not chunk:\n                            break\n                        yield chunk\n            return foo_reader, None\n        mpv.play('python://foo23')\n        mpv.wait_for_playback()\n        catchall.unregister()\n        "
        if self._python_stream_catchall is not None:
            raise KeyError('A catch-all python stream is already registered')
        self._python_stream_catchall = cb

        def unregister():
            if False:
                for i in range(10):
                    print('nop')
            if self._python_stream_catchall is not cb:
                raise RuntimeError('This catch-all python stream has already been unregistered')
            self._python_stream_catchall = None
        cb.unregister = unregister
        return cb

    def _get_property(self, name, decoder=strict_decoder, fmt=MpvFormat.NODE):
        if False:
            while True:
                i = 10
        self.check_core_alive()
        out = create_string_buffer(sizeof(MpvNode))
        try:
            cval = _mpv_get_property(self.handle, name.encode('utf-8'), fmt, out)
            if fmt is MpvFormat.OSD_STRING:
                return cast(out, POINTER(c_char_p)).contents.value.decode('utf-8')
            elif fmt is MpvFormat.NODE:
                rv = cast(out, POINTER(MpvNode)).contents.node_value(decoder=decoder)
                _mpv_free_node_contents(out)
                return rv
            else:
                raise TypeError('_get_property only supports NODE and OSD_STRING formats.')
        except PropertyUnavailableError as ex:
            return None

    def _set_property(self, name, value):
        if False:
            i = 10
            return i + 15
        self.check_core_alive()
        ename = name.encode('utf-8')
        if isinstance(value, (list, set, dict)):
            (_1, _2, _3, pointer) = _make_node_str_list(value)
            _mpv_set_property(self.handle, ename, MpvFormat.NODE, pointer)
        else:
            _mpv_set_property_string(self.handle, ename, _mpv_coax_proptype(value))

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        return self._get_property(_py_to_mpv(name), lazy_decoder)

    def __setattr__(self, name, value):
        if False:
            return 10
        try:
            if name != 'handle' and (not name.startswith('_')):
                self._set_property(_py_to_mpv(name), value)
            else:
                super().__setattr__(name, value)
        except AttributeError:
            super().__setattr__(name, value)

    def __dir__(self):
        if False:
            return 10
        return super().__dir__() + [name.replace('-', '_') for name in self.property_list]

    @property
    def properties(self):
        if False:
            for i in range(10):
                print('nop')
        return {name: self.option_info(name) for name in self.property_list}

    def __getitem__(self, name, file_local=False):
        if False:
            print('Hello World!')
        'Get an option value.'
        prefix = 'file-local-options/' if file_local else 'options/'
        return self._get_property(prefix + name, lazy_decoder)

    def __setitem__(self, name, value, file_local=False):
        if False:
            print('Hello World!')
        'Set an option value.'
        prefix = 'file-local-options/' if file_local else 'options/'
        return self._set_property(prefix + name, value)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        'Iterate over all option names.'
        return iter(self.options)

    def option_info(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Get information on the given option.'
        try:
            return self._get_property('option-info/' + name)
        except AttributeError:
            return None

class MpvRenderContext:

    def __init__(self, mpv, api_type, **kwargs):
        if False:
            i = 10
            return i + 15
        self._mpv = mpv
        kwargs['api_type'] = api_type
        buf = cast(create_string_buffer(sizeof(MpvRenderCtxHandle)), POINTER(MpvRenderCtxHandle))
        _mpv_render_context_create(buf, mpv.handle, kwargs_to_render_param_array(kwargs))
        self._handle = buf.contents

    def free(self):
        if False:
            for i in range(10):
                print('nop')
        _mpv_render_context_free(self._handle)

    def __setattr__(self, name, value):
        if False:
            i = 10
            return i + 15
        if name.startswith('_'):
            super().__setattr__(name, value)
        elif name == 'update_cb':
            func = value if value else lambda : None
            self._update_cb = value
            self._update_fn_wrapper = RenderUpdateFn(lambda _userdata: func())
            _mpv_render_context_set_update_callback(self._handle, self._update_fn_wrapper, None)
        else:
            param = MpvRenderParam(name, value)
            _mpv_render_context_set_parameter(self._handle, param)

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        if name == 'update_cb':
            return self._update_cb
        elif name == 'handle':
            return self._handle
        param = MpvRenderParam(name)
        data_type = type(param.data.contents)
        buf = cast(create_string_buffer(sizeof(data_type)), POINTER(data_type))
        param.data = buf
        _mpv_render_context_get_info(self._handle, param)
        return buf.contents.as_dict()

    def update(self):
        if False:
            return 10
        ' Calls mpv_render_context_update and returns the MPV_RENDER_UPDATE_FRAME flag (see render.h) '
        return bool(_mpv_render_context_update(self._handle) & 1)

    def render(self, **kwargs):
        if False:
            return 10
        _mpv_render_context_render(self._handle, kwargs_to_render_param_array(kwargs))

    def report_swap(self):
        if False:
            for i in range(10):
                print('nop')
        _mpv_render_context_report_swap(self._handle)