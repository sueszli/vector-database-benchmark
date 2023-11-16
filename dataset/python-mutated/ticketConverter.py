import argparse
import struct
from impacket import version
from impacket.krb5.ccache import CCache

def parse_args():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='File in kirbi (KRB-CRED) or ccache format')
    parser.add_argument('output_file', help='Output file')
    return parser.parse_args()

def main():
    if False:
        return 10
    print(version.BANNER)
    args = parse_args()
    if is_kirbi_file(args.input_file):
        print('[*] converting kirbi to ccache...')
        convert_kirbi_to_ccache(args.input_file, args.output_file)
        print('[+] done')
    elif is_ccache_file(args.input_file):
        print('[*] converting ccache to kirbi...')
        convert_ccache_to_kirbi(args.input_file, args.output_file)
        print('[+] done')
    else:
        print('[X] unknown file format')

def is_kirbi_file(filename):
    if False:
        i = 10
        return i + 15
    with open(filename, 'rb') as fi:
        fileid = struct.unpack('>B', fi.read(1))[0]
    return fileid == 118

def is_ccache_file(filename):
    if False:
        return 10
    with open(filename, 'rb') as fi:
        fileid = struct.unpack('>B', fi.read(1))[0]
    return fileid == 5

def convert_kirbi_to_ccache(input_filename, output_filename):
    if False:
        i = 10
        return i + 15
    ccache = CCache.loadKirbiFile(input_filename)
    ccache.saveFile(output_filename)

def convert_ccache_to_kirbi(input_filename, output_filename):
    if False:
        print('Hello World!')
    ccache = CCache.loadFile(input_filename)
    ccache.saveKirbiFile(output_filename)
if __name__ == '__main__':
    main()