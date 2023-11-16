"""This module implements enough functionality to program the STM32F4xx over
DFU, without requiring dfu-util.

See app note AN3156 for a description of the DFU protocol.
See document UM0391 for a description of the DFuse file.
"""
from __future__ import print_function
import argparse
import collections
import inspect
import re
import struct
import sys
import usb.core
import usb.util
import zlib
__TIMEOUT = 4000
__DFU_DETACH = 0
__DFU_DNLOAD = 1
__DFU_UPLOAD = 2
__DFU_GETSTATUS = 3
__DFU_CLRSTATUS = 4
__DFU_GETSTATE = 5
__DFU_ABORT = 6
__DFU_STATE_APP_IDLE = 0
__DFU_STATE_APP_DETACH = 1
__DFU_STATE_DFU_IDLE = 2
__DFU_STATE_DFU_DOWNLOAD_SYNC = 3
__DFU_STATE_DFU_DOWNLOAD_BUSY = 4
__DFU_STATE_DFU_DOWNLOAD_IDLE = 5
__DFU_STATE_DFU_MANIFEST_SYNC = 6
__DFU_STATE_DFU_MANIFEST = 7
__DFU_STATE_DFU_MANIFEST_WAIT_RESET = 8
__DFU_STATE_DFU_UPLOAD_IDLE = 9
__DFU_STATE_DFU_ERROR = 10
_DFU_DESCRIPTOR_TYPE = 33
__DFU_STATUS_STR = {__DFU_STATE_APP_IDLE: 'STATE_APP_IDLE', __DFU_STATE_APP_DETACH: 'STATE_APP_DETACH', __DFU_STATE_DFU_IDLE: 'STATE_DFU_IDLE', __DFU_STATE_DFU_DOWNLOAD_SYNC: 'STATE_DFU_DOWNLOAD_SYNC', __DFU_STATE_DFU_DOWNLOAD_BUSY: 'STATE_DFU_DOWNLOAD_BUSY', __DFU_STATE_DFU_DOWNLOAD_IDLE: 'STATE_DFU_DOWNLOAD_IDLE', __DFU_STATE_DFU_MANIFEST_SYNC: 'STATE_DFU_MANIFEST_SYNC', __DFU_STATE_DFU_MANIFEST: 'STATE_DFU_MANIFEST', __DFU_STATE_DFU_MANIFEST_WAIT_RESET: 'STATE_DFU_MANIFEST_WAIT_RESET', __DFU_STATE_DFU_UPLOAD_IDLE: 'STATE_DFU_UPLOAD_IDLE', __DFU_STATE_DFU_ERROR: 'STATE_DFU_ERROR'}
__dev = None
__cfg_descr = None
__verbose = None
__DFU_INTERFACE = 0
getargspec = getattr(inspect, 'getfullargspec', getattr(inspect, 'getargspec', None))
if 'length' in getargspec(usb.util.get_string).args:

    def get_string(dev, index):
        if False:
            print('Hello World!')
        return usb.util.get_string(dev, 255, index)
else:

    def get_string(dev, index):
        if False:
            for i in range(10):
                print('nop')
        return usb.util.get_string(dev, index)

def find_dfu_cfg_descr(descr):
    if False:
        while True:
            i = 10
    if len(descr) == 9 and descr[0] == 9 and (descr[1] == _DFU_DESCRIPTOR_TYPE):
        nt = collections.namedtuple('CfgDescr', ['bLength', 'bDescriptorType', 'bmAttributes', 'wDetachTimeOut', 'wTransferSize', 'bcdDFUVersion'])
        return nt(*struct.unpack('<BBBHHH', bytearray(descr)))
    return None

def init(**kwargs):
    if False:
        print('Hello World!')
    'Initializes the found DFU device so that we can program it.'
    global __dev, __cfg_descr
    devices = get_dfu_devices(**kwargs)
    if not devices:
        raise ValueError('No DFU device found')
    if len(devices) > 1:
        raise ValueError('Multiple DFU devices found')
    __dev = devices[0]
    __dev.set_configuration()
    usb.util.claim_interface(__dev, __DFU_INTERFACE)
    __cfg_descr = None
    for cfg in __dev.configurations():
        __cfg_descr = find_dfu_cfg_descr(cfg.extra_descriptors)
        if __cfg_descr:
            break
        for itf in cfg.interfaces():
            __cfg_descr = find_dfu_cfg_descr(itf.extra_descriptors)
            if __cfg_descr:
                break
    for attempt in range(4):
        status = get_status()
        if status == __DFU_STATE_DFU_IDLE:
            break
        elif status == __DFU_STATE_DFU_DOWNLOAD_IDLE or status == __DFU_STATE_DFU_UPLOAD_IDLE:
            abort_request()
        else:
            clr_status()

def abort_request():
    if False:
        for i in range(10):
            print('nop')
    'Sends an abort request.'
    __dev.ctrl_transfer(33, __DFU_ABORT, 0, __DFU_INTERFACE, None, __TIMEOUT)

def clr_status():
    if False:
        i = 10
        return i + 15
    'Clears any error status (perhaps left over from a previous session).'
    __dev.ctrl_transfer(33, __DFU_CLRSTATUS, 0, __DFU_INTERFACE, None, __TIMEOUT)

def get_status():
    if False:
        for i in range(10):
            print('nop')
    'Get the status of the last operation.'
    stat = __dev.ctrl_transfer(161, __DFU_GETSTATUS, 0, __DFU_INTERFACE, 6, 20000)
    if stat[5]:
        message = get_string(__dev, stat[5])
        if message:
            print(message)
    return stat[4]

def check_status(stage, expected):
    if False:
        i = 10
        return i + 15
    status = get_status()
    if status != expected:
        raise SystemExit('DFU: %s failed (%s)' % (stage, __DFU_STATUS_STR.get(status, status)))

def mass_erase():
    if False:
        for i in range(10):
            print('nop')
    'Performs a MASS erase (i.e. erases the entire device).'
    __dev.ctrl_transfer(33, __DFU_DNLOAD, 0, __DFU_INTERFACE, 'A', __TIMEOUT)
    check_status('erase', __DFU_STATE_DFU_DOWNLOAD_BUSY)
    check_status('erase', __DFU_STATE_DFU_DOWNLOAD_IDLE)

def page_erase(addr):
    if False:
        i = 10
        return i + 15
    'Erases a single page.'
    if __verbose:
        print('Erasing page: 0x%x...' % addr)
    buf = struct.pack('<BI', 65, addr)
    __dev.ctrl_transfer(33, __DFU_DNLOAD, 0, __DFU_INTERFACE, buf, __TIMEOUT)
    check_status('erase', __DFU_STATE_DFU_DOWNLOAD_BUSY)
    check_status('erase', __DFU_STATE_DFU_DOWNLOAD_IDLE)

def set_address(addr):
    if False:
        for i in range(10):
            print('nop')
    'Sets the address for the next operation.'
    buf = struct.pack('<BI', 33, addr)
    __dev.ctrl_transfer(33, __DFU_DNLOAD, 0, __DFU_INTERFACE, buf, __TIMEOUT)
    check_status('set address', __DFU_STATE_DFU_DOWNLOAD_BUSY)
    check_status('set address', __DFU_STATE_DFU_DOWNLOAD_IDLE)

def write_memory(addr, buf, progress=None, progress_addr=0, progress_size=0):
    if False:
        i = 10
        return i + 15
    'Writes a buffer into memory. This routine assumes that memory has\n    already been erased.\n    '
    xfer_count = 0
    xfer_bytes = 0
    xfer_total = len(buf)
    xfer_base = addr
    while xfer_bytes < xfer_total:
        if __verbose and xfer_count % 512 == 0:
            print('Addr 0x%x %dKBs/%dKBs...' % (xfer_base + xfer_bytes, xfer_bytes // 1024, xfer_total // 1024))
        if progress and xfer_count % 2 == 0:
            progress(progress_addr, xfer_base + xfer_bytes - progress_addr, progress_size)
        set_address(xfer_base + xfer_bytes)
        chunk = min(__cfg_descr.wTransferSize, xfer_total - xfer_bytes)
        __dev.ctrl_transfer(33, __DFU_DNLOAD, 2, __DFU_INTERFACE, buf[xfer_bytes:xfer_bytes + chunk], __TIMEOUT)
        check_status('write memory', __DFU_STATE_DFU_DOWNLOAD_BUSY)
        check_status('write memory', __DFU_STATE_DFU_DOWNLOAD_IDLE)
        xfer_count += 1
        xfer_bytes += chunk

def write_page(buf, xfer_offset):
    if False:
        while True:
            i = 10
    'Writes a single page. This routine assumes that memory has already\n    been erased.\n    '
    xfer_base = 134217728
    set_address(xfer_base + xfer_offset)
    __dev.ctrl_transfer(33, __DFU_DNLOAD, 2, __DFU_INTERFACE, buf, __TIMEOUT)
    check_status('write memory', __DFU_STATE_DFU_DOWNLOAD_BUSY)
    check_status('write memory', __DFU_STATE_DFU_DOWNLOAD_IDLE)
    if __verbose:
        print('Write: 0x%x ' % (xfer_base + xfer_offset))

def exit_dfu():
    if False:
        return 10
    'Exit DFU mode, and start running the program.'
    set_address(134217728)
    __dev.ctrl_transfer(33, __DFU_DNLOAD, 0, __DFU_INTERFACE, None, __TIMEOUT)
    try:
        if get_status() != __DFU_STATE_DFU_MANIFEST:
            print('Failed to reset device')
        usb.util.dispose_resources(__dev)
    except:
        pass

def named(values, names):
    if False:
        for i in range(10):
            print('nop')
    'Creates a dict with `names` as fields, and `values` as values.'
    return dict(zip(names.split(), values))

def consume(fmt, data, names):
    if False:
        print('Hello World!')
    'Parses the struct defined by `fmt` from `data`, stores the parsed fields\n    into a named tuple using `names`. Returns the named tuple, and the data\n    with the struct stripped off.'
    size = struct.calcsize(fmt)
    return (named(struct.unpack(fmt, data[:size]), names), data[size:])

def cstring(string):
    if False:
        while True:
            i = 10
    'Extracts a null-terminated string from a byte array.'
    return string.decode('utf-8').split('\x00', 1)[0]

def compute_crc(data):
    if False:
        while True:
            i = 10
    'Computes the CRC32 value for the data passed in.'
    return 4294967295 & -zlib.crc32(data) - 1

def read_dfu_file(filename):
    if False:
        print('Hello World!')
    'Reads a DFU file, and parses the individual elements from the file.\n    Returns an array of elements. Each element is a dictionary with the\n    following keys:\n        num     - The element index.\n        address - The address that the element data should be written to.\n        size    - The size of the element data.\n        data    - The element data.\n    If an error occurs while parsing the file, then None is returned.\n    '
    print('File: {}'.format(filename))
    with open(filename, 'rb') as fin:
        data = fin.read()
    crc = compute_crc(data[:-4])
    elements = []
    (dfu_prefix, data) = consume('<5sBIB', data, 'signature version size targets')
    print('    %(signature)s v%(version)d, image size: %(size)d, targets: %(targets)d' % dfu_prefix)
    for target_idx in range(dfu_prefix['targets']):
        (img_prefix, data) = consume('<6sBI255s2I', data, 'signature altsetting named name size elements')
        img_prefix['num'] = target_idx
        if img_prefix['named']:
            img_prefix['name'] = cstring(img_prefix['name'])
        else:
            img_prefix['name'] = ''
        print('    %(signature)s %(num)d, alt setting: %(altsetting)s, name: "%(name)s", size: %(size)d, elements: %(elements)d' % img_prefix)
        target_size = img_prefix['size']
        target_data = data[:target_size]
        data = data[target_size:]
        for elem_idx in range(img_prefix['elements']):
            (elem_prefix, target_data) = consume('<2I', target_data, 'addr size')
            elem_prefix['num'] = elem_idx
            print('      %(num)d, address: 0x%(addr)08x, size: %(size)d' % elem_prefix)
            elem_size = elem_prefix['size']
            elem_data = target_data[:elem_size]
            target_data = target_data[elem_size:]
            elem_prefix['data'] = elem_data
            elements.append(elem_prefix)
        if len(target_data):
            print('target %d PARSE ERROR' % target_idx)
    dfu_suffix = named(struct.unpack('<4H3sBI', data[:16]), 'device product vendor dfu ufd len crc')
    print('    usb: %(vendor)04x:%(product)04x, device: 0x%(device)04x, dfu: 0x%(dfu)04x, %(ufd)s, %(len)d, 0x%(crc)08x' % dfu_suffix)
    if crc != dfu_suffix['crc']:
        print('CRC ERROR: computed crc32 is 0x%08x' % crc)
        return
    data = data[16:]
    if data:
        print('PARSE ERROR')
        return
    return elements

class FilterDFU(object):
    """Class for filtering USB devices to identify devices which are in DFU
    mode.
    """

    def __call__(self, device):
        if False:
            i = 10
            return i + 15
        for cfg in device:
            for intf in cfg:
                return intf.bInterfaceClass == 254 and intf.bInterfaceSubClass == 1

def get_dfu_devices(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of USB devices which are currently in DFU mode.\n    Additional filters (like idProduct and idVendor) can be passed in\n    to refine the search.\n    '
    return list(usb.core.find(*args, find_all=True, custom_match=FilterDFU(), **kwargs))

def get_memory_layout(device):
    if False:
        return 10
    'Returns an array which identifies the memory layout. Each entry\n    of the array will contain a dictionary with the following keys:\n        addr        - Address of this memory segment.\n        last_addr   - Last address contained within the memory segment.\n        size        - Size of the segment, in bytes.\n        num_pages   - Number of pages in the segment.\n        page_size   - Size of each page, in bytes.\n    '
    cfg = device[0]
    intf = cfg[0, 0]
    mem_layout_str = get_string(device, intf.iInterface)
    mem_layout = mem_layout_str.split('/')
    result = []
    for mem_layout_index in range(1, len(mem_layout), 2):
        addr = int(mem_layout[mem_layout_index], 0)
        segments = mem_layout[mem_layout_index + 1].split(',')
        seg_re = re.compile('(\\d+)\\*(\\d+)(.)(.)')
        for segment in segments:
            seg_match = seg_re.match(segment)
            num_pages = int(seg_match.groups()[0], 10)
            page_size = int(seg_match.groups()[1], 10)
            multiplier = seg_match.groups()[2]
            if multiplier == 'K':
                page_size *= 1024
            if multiplier == 'M':
                page_size *= 1024 * 1024
            size = num_pages * page_size
            last_addr = addr + size - 1
            result.append(named((addr, last_addr, size, num_pages, page_size), 'addr last_addr size num_pages page_size'))
            addr += size
    return result

def list_dfu_devices(*args, **kwargs):
    if False:
        print('Hello World!')
    'Prints a list of devices detected in DFU mode.'
    devices = get_dfu_devices(*args, **kwargs)
    if not devices:
        raise SystemExit('No DFU capable devices found')
    for device in devices:
        print('Bus {} Device {:03d}: ID {:04x}:{:04x}'.format(device.bus, device.address, device.idVendor, device.idProduct))
        layout = get_memory_layout(device)
        print('Memory Layout')
        for entry in layout:
            print('    0x{:x} {:2d} pages of {:3d}K bytes'.format(entry['addr'], entry['num_pages'], entry['page_size'] // 1024))

def write_elements(elements, mass_erase_used, progress=None):
    if False:
        while True:
            i = 10
    'Writes the indicated elements into the target memory,\n    erasing as needed.\n    '
    mem_layout = get_memory_layout(__dev)
    for elem in elements:
        addr = elem['addr']
        size = elem['size']
        data = elem['data']
        elem_size = size
        elem_addr = addr
        if progress and elem_size:
            progress(elem_addr, 0, elem_size)
        while size > 0:
            write_size = size
            if not mass_erase_used:
                for segment in mem_layout:
                    if addr >= segment['addr'] and addr <= segment['last_addr']:
                        page_size = segment['page_size']
                        page_addr = addr & ~(page_size - 1)
                        if addr + write_size > page_addr + page_size:
                            write_size = page_addr + page_size - addr
                        page_erase(page_addr)
                        break
            write_memory(addr, data[:write_size], progress, elem_addr, elem_size)
            data = data[write_size:]
            addr += write_size
            size -= write_size
            if progress:
                progress(elem_addr, addr - elem_addr, elem_size)

def cli_progress(addr, offset, size):
    if False:
        for i in range(10):
            print('nop')
    'Prints a progress report suitable for use on the command line.'
    width = 25
    done = offset * width // size
    print('\r0x{:08x} {:7d} [{}{}] {:3d}% '.format(addr, size, '=' * done, ' ' * (width - done), offset * 100 // size), end='')
    try:
        sys.stdout.flush()
    except OSError:
        pass
    if offset == size:
        print('')

def main():
    if False:
        i = 10
        return i + 15
    'Test program for verifying this files functionality.'
    global __verbose
    parser = argparse.ArgumentParser(description='DFU Python Util')
    parser.add_argument('-l', '--list', help='list available DFU devices', action='store_true', default=False)
    parser.add_argument('--vid', help='USB Vendor ID', type=lambda x: int(x, 0), default=None)
    parser.add_argument('--pid', help='USB Product ID', type=lambda x: int(x, 0), default=None)
    parser.add_argument('-m', '--mass-erase', help='mass erase device', action='store_true', default=False)
    parser.add_argument('-u', '--upload', help='read file from DFU device', dest='path', default=False)
    parser.add_argument('-x', '--exit', help='Exit DFU', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true', default=False)
    args = parser.parse_args()
    __verbose = args.verbose
    kwargs = {}
    if args.vid:
        kwargs['idVendor'] = args.vid
    if args.pid:
        kwargs['idProduct'] = args.pid
    if args.list:
        list_dfu_devices(**kwargs)
        return
    init(**kwargs)
    command_run = False
    if args.mass_erase:
        print('Mass erase...')
        mass_erase()
        command_run = True
    if args.path:
        elements = read_dfu_file(args.path)
        if not elements:
            print('No data in dfu file')
            return
        print('Writing memory...')
        write_elements(elements, args.mass_erase, progress=cli_progress)
        print('Exiting DFU...')
        exit_dfu()
        command_run = True
    if args.exit:
        print('Exiting DFU...')
        exit_dfu()
        command_run = True
    if command_run:
        print('Finished')
    else:
        print('No command specified')
if __name__ == '__main__':
    main()