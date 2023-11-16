import glob
import struct
import sys
IMAGE_FILE_MACHINE_AMD64 = 34404
IMAGE_FILE_MACHINE_I386 = 332
IMAGE_FILE_MACHINE_IA64 = 512

def check_pefile(filename):
    if False:
        i = 10
        return i + 15
    with open(filename, 'rb') as fh:
        s = fh.read(2)
        if s != b'MZ':
            return (None, 'Not an PE file')
        else:
            fh.seek(60)
            s = fh.read(4)
            header_offset = struct.unpack('<L', s)[0]
            fh.seek(header_offset + 4)
            s = fh.read(2)
            machine = struct.unpack('<H', s)[0]
    if machine == IMAGE_FILE_MACHINE_I386:
        return (32, 'IA-32 (32-bit x86)')
    elif machine == IMAGE_FILE_MACHINE_IA64:
        return (64, 'IA-64 (Itanium)')
    elif machine == IMAGE_FILE_MACHINE_AMD64:
        return (64, 'AMD64 (64-bit x86)')
    else:
        return (None, 'Handled architecture: 0x%x' % machine)

def check(filename, expected_bits):
    if False:
        for i in range(10):
            print('nop')
    (bits, desc) = check_pefile(filename)
    okay = True
    msg = '** okay  '
    if bits != expected_bits:
        msg = '** FAILED'
        okay = False
    print(msg, filename, desc, sep='\t')
    return okay

def main():
    if False:
        return 10
    expected_bits = int(sys.argv[1])
    okay = True
    for pat in sys.argv[2:]:
        filenames = glob.glob(pat)
        for filename in filenames:
            okay = check(filename, expected_bits) and okay
    if not okay:
        raise SystemExit('*** FAILED.')
    else:
        print('*** Okay.')
if __name__ == '__main__':
    main()