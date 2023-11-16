import os
import re
import struct
import sys
from binascii import crc32
from rfcore_firmware import validate_crc, _OBFUSCATION_KEY
_FIRMWARE_FILES = {'stm32wb5x_FUS_fw_1_0_2.bin': 'fus_102.bin', 'stm32wb5x_FUS_fw.bin': 'fus_110.bin', 'stm32wb5x_BLE_HCILayer_fw.bin': 'ws_ble_hci.bin'}
_RELEASE_NOTES = 'Release_Notes.html'

def get_details(release_notes, filename):
    if False:
        for i in range(10):
            print('nop')
    if not release_notes:
        return None
    file_details = re.findall(b'%s,(((0x[\\d\\S]+?),)+[vV][\\d\\.]+)[<,]' % filename.encode(), release_notes, flags=re.DOTALL)
    latest_details = file_details[0][0].split(b',')
    (addr_1m, addr_640k, addr_512k, addr_256k, version) = latest_details
    addr_1m = int(addr_1m, 0)
    addr_640k = int(addr_640k, 0)
    addr_512k = int(addr_512k, 0)
    addr_256k = int(addr_256k, 0)
    version = [int(v) for v in version.lower().lstrip(b'v').split(b'.')]
    return (addr_1m, addr_640k, addr_512k, addr_256k, version)

def main(src_path, dest_path):
    if False:
        while True:
            i = 10
    with open(os.path.join(src_path, _RELEASE_NOTES), 'rb') as f:
        release_notes = f.read()
    release_notes = re.sub(b'</?strong>', b'', release_notes)
    release_notes = re.sub(b'</t[dh]>\\W*\\n*\\W*', b',', release_notes.replace(b'<td>', b'').replace(b'<th>', b''))
    if b'Wireless Coprocessor Binary,STM32WB5xxG(1M),STM32WB5xxY(640k),STM32WB5xxE(512K),STM32WB5xxC(256K),Version,' not in release_notes:
        raise SystemExit('Cannot determine binary load address, please confirm Coprocessor folder / Release Notes format.')
    for (src_filename, dest_file) in _FIRMWARE_FILES.items():
        src_file = os.path.join(src_path, src_filename)
        dest_file = os.path.join(dest_path, dest_file)
        if not os.path.exists(src_file):
            print('Unable to find: {}'.format(src_file))
            continue
        sz = 0
        with open(src_file, 'rb') as src:
            crc = 0
            with open(dest_file, 'wb') as dest:
                while True:
                    b = src.read(4)
                    if not b:
                        break
                    (v,) = struct.unpack('<I', b)
                    v ^= _OBFUSCATION_KEY
                    vs = struct.pack('<I', v)
                    dest.write(vs)
                    crc = crc32(vs, crc)
                    sz += 4
                (addr_1m, addr_640k, addr_512k, addr_256k, version) = get_details(release_notes, src_filename)
                footer = struct.pack('<37sIIIIbbbI', src_filename.encode(), addr_1m, addr_640k, addr_512k, addr_256k, *version, _OBFUSCATION_KEY)
                assert len(footer) == 60
                dest.write(footer)
                crc = crc32(footer, crc)
                crc = 4294967295 & -crc - 1
                dest.write(struct.pack('<I', crc))
                sz += 64
        print(f'Written {src_filename} v{version[0]}.{version[1]}.{version[2]} to {dest_file} ({sz} bytes)')
        check_file_details(dest_file)

def check_file_details(filename):
    if False:
        return 10
    'Should match copy of function in rfcore_firmware.py to confirm operation'
    with open(filename, 'rb') as f:
        if not validate_crc(f):
            raise ValueError('file validation failed: incorrect crc')
        f.seek(-64, 2)
        footer = f.read()
        assert len(footer) == 64
        details = struct.unpack('<37sIIIIbbbII', footer)
        (src_filename, addr_1m, addr_640k, addr_512k, addr_256k, vers_major, vers_minor, vers_patch, KEY, crc) = details
        if KEY != _OBFUSCATION_KEY:
            raise ValueError('file validation failed: incorrect key')
    return (src_filename, addr_1m, addr_640k, addr_512k, addr_256k, (vers_major, vers_minor, vers_patch))
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: {} src_path dest_path'.format(sys.argv[0]))
        print()
        print('"src_path" should be the location of the ST binaries from https://github.com/STMicroelectronics/STM32CubeWB/tree/master/Projects/STM32WB_Copro_Wireless_Binaries/STM32WB5x')
        print('"dest_path" will be where fus_102.bin, fus_110.bin, and ws_ble_hci.bin will be written to.')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])