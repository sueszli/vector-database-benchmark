import os
import sys
import ctypes
from typing import Optional, Mapping
from .util import UserFacingException
from .i18n import _
from .logging import get_logger
_logger = get_logger(__name__)
if sys.platform == 'darwin':
    name = 'libzbar.0.dylib'
elif sys.platform in ('windows', 'win32'):
    name = 'libzbar-0.dll'
else:
    name = 'libzbar.so.0'
try:
    libzbar = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), name))
except BaseException as e1:
    try:
        libzbar = ctypes.cdll.LoadLibrary(name)
    except BaseException as e2:
        libzbar = None
        _logger.error(f'failed to load zbar. exceptions: {[e1, e2]!r}')

def scan_barcode(device='', timeout=-1, display=True, threaded=False) -> Optional[str]:
    if False:
        return 10
    if libzbar is None:
        raise UserFacingException('Cannot start QR scanner: zbar not available.')
    libzbar.zbar_symbol_get_data.restype = ctypes.c_char_p
    libzbar.zbar_processor_create.restype = ctypes.POINTER(ctypes.c_int)
    libzbar.zbar_processor_get_results.restype = ctypes.POINTER(ctypes.c_int)
    libzbar.zbar_symbol_set_first_symbol.restype = ctypes.POINTER(ctypes.c_int)
    proc = libzbar.zbar_processor_create(threaded)
    libzbar.zbar_processor_request_size(proc, 640, 480)
    if libzbar.zbar_processor_init(proc, device.encode('utf-8'), display) != 0:
        raise UserFacingException(_('Cannot start QR scanner: initialization failed.') + '\n' + _('Make sure you have a camera connected and enabled.'))
    libzbar.zbar_processor_set_visible(proc)
    if libzbar.zbar_process_one(proc, timeout):
        symbols = libzbar.zbar_processor_get_results(proc)
    else:
        symbols = None
    libzbar.zbar_processor_destroy(proc)
    if symbols is None:
        return
    if not libzbar.zbar_symbol_set_get_size(symbols):
        return
    symbol = libzbar.zbar_symbol_set_first_symbol(symbols)
    data = libzbar.zbar_symbol_get_data(symbol)
    return data.decode('utf8')

def find_system_cameras() -> Mapping[str, str]:
    if False:
        for i in range(10):
            print('nop')
    device_root = '/sys/class/video4linux'
    devices = {}
    if os.path.exists(device_root):
        for device in os.listdir(device_root):
            path = os.path.join(device_root, device, 'name')
            try:
                with open(path, encoding='utf-8') as f:
                    name = f.read()
            except Exception:
                continue
            name = name.strip('\n')
            devices[name] = os.path.join('/dev', device)
    return devices

def version_info() -> Mapping[str, Optional[str]]:
    if False:
        for i in range(10):
            print('nop')
    return {'libzbar.path': libzbar._name if libzbar else None}
if __name__ == '__main__':
    print(scan_barcode())