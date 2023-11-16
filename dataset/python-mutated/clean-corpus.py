import sys
import os

def is_elf(file):
    if False:
        i = 10
        return i + 15
    magic = None
    with open(file, 'rb') as f:
        raw = f.read()
        magic = raw[:4]
    return magic[0] == 127 and magic[1] == ord('E') and (magic[2] == ord('L')) and (magic[3] == ord('F'))

def is_pe(file):
    if False:
        for i in range(10):
            print('nop')
    magic = None
    with open(file, 'rb') as f:
        raw = f.read()
        magic = raw[:2]
    return magic[0] == ord('M') and magic[1] == ord('Z')

def is_macho(file):
    if False:
        return 10
    magic = None
    with open(file, 'rb') as f:
        raw = f.read()
        magic = raw[:4]
    magic = list(magic)
    magics = [[254, 237, 250, 206], [206, 250, 237, 254], [254, 237, 250, 207], [207, 250, 237, 254], [202, 254, 186, 190], [190, 186, 254, 202]]
    return any((m == magic for m in magics))

def clean(directory):
    if False:
        for i in range(10):
            print('nop')
    whitelist = ['.git']
    for (dirname, subdir, files) in os.walk(directory):
        if any((d in dirname for d in whitelist)):
            continue
        for f in files:
            fullpath = os.path.join(dirname, f)
            if not (is_elf(fullpath) or is_pe(fullpath) or is_macho(fullpath)):
                print("Removing '{}'".format(fullpath))
                try:
                    os.remove(fullpath)
                except Exception as e:
                    print('Error: {}'.format(e))

def main():
    if False:
        for i in range(10):
            print('nop')
    if len(sys.argv) != 2:
        print('Usage: {} <corpus>'.format(sys.argv[0]))
        return 1
    clean(sys.argv[1])
    return 0
if __name__ == '__main__':
    main()