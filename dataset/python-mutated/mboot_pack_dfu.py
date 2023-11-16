"""
Utility to create compressed, encrypted and signed DFU files.
"""
import argparse
import os
import re
import struct
import sys
import zlib
sys.path.append(os.path.dirname(__file__) + '/../../../tools')
import dfu
try:
    import pyhy
except ImportError:
    raise SystemExit('ERROR: pyhy not found. Please install python pyhy for encrypted mboot support: pip3 install pyhy')
MBOOT_PACK_HEADER_VERSION = 1
MBOOT_PACK_HYDRO_CONTEXT = 'mbootenc'
MBOOT_PACK_CHUNK_META = 0
MBOOT_PACK_CHUNK_FULL_SIG = 1
MBOOT_PACK_CHUNK_FW_RAW = 2
MBOOT_PACK_CHUNK_FW_GZIP = 3

class Keys:

    def __init__(self, filename):
        if False:
            for i in range(10):
                print('nop')
        self.filename = filename

    def generate(self):
        if False:
            while True:
                i = 10
        kp = pyhy.hydro_sign_keygen()
        self.sign_sk = kp.sk
        self.sign_pk = kp.pk
        self.secretbox = pyhy.hydro_secretbox_keygen()

    def _save_data(self, name, data, file_, hide=False):
        if False:
            while True:
                i = 10
        prefix = '//' if hide else ''
        data = ','.join(('0x{:02x}'.format(b) for b in data))
        file_.write('{}const uint8_t {}[] = {{{}}};\n'.format(prefix, name, data))

    def _load_data(self, name, line):
        if False:
            i = 10
            return i + 15
        line = line.split(name + '[] = ')
        if len(line) != 2:
            raise Exception('malformed input keys: {}'.format(line))
        data = line[1].strip()
        return bytes((int(value, 16) for value in data[1:-2].split(',')))

    def save(self):
        if False:
            while True:
                i = 10
        with open(self.filename, 'w') as f:
            self._save_data('mboot_pack_sign_secret_key', self.sign_sk, f, hide=True)
            self._save_data('mboot_pack_sign_public_key', self.sign_pk, f)
            self._save_data('mboot_pack_secretbox_key', self.secretbox, f)

    def load(self):
        if False:
            i = 10
            return i + 15
        with open(self.filename) as f:
            for line in f:
                for (key, attr) in (('mboot_pack_sign_secret_key', 'sign_sk'), ('mboot_pack_sign_public_key', 'sign_pk'), ('mboot_pack_secretbox_key', 'secretbox')):
                    if key in line:
                        setattr(self, attr, self._load_data(key, line))

def dfu_read(filename):
    if False:
        print('Hello World!')
    elems = []
    with open(filename, 'rb') as f:
        hdr = f.read(11)
        (sig, ver, size, num_targ) = struct.unpack('<5sBIB', hdr)
        file_offset = 11
        for i in range(num_targ):
            hdr = f.read(274)
            (sig, alt, has_name, name, t_size, num_elem) = struct.unpack('<6sBi255sII', hdr)
            file_offset += 274
            file_offset_t = file_offset
            for j in range(num_elem):
                hdr = f.read(8)
                (addr, e_size) = struct.unpack('<II', hdr)
                data = f.read(e_size)
                elems.append((addr, data))
                file_offset += 8 + e_size
            if t_size != file_offset - file_offset_t:
                raise Exception('corrupt DFU {} {}'.format(t_size, file_offset - file_offset_t))
        if size != file_offset:
            raise Exception('corrupt DFU {} {}'.format(size, file_offset))
        hdr = f.read(16)
        hdr = struct.unpack('<HHHH3sBI', hdr)
        vid_pid = '0x{:04x}:0x{:04x}'.format(hdr[2], hdr[1])
    return (vid_pid, elems)

def compress(data):
    if False:
        return 10
    c = zlib.compressobj(level=9, memLevel=9, wbits=-15)
    return c.compress(data) + c.flush()

def encrypt(keys, data):
    if False:
        return 10
    return pyhy.hydro_secretbox_encrypt(data, 0, MBOOT_PACK_HYDRO_CONTEXT, keys.secretbox)

def sign(keys, data):
    if False:
        i = 10
        return i + 15
    if not hasattr(keys, 'sign_sk'):
        raise Exception('packing a dfu requires a secret key')
    return pyhy.hydro_sign_create(data, MBOOT_PACK_HYDRO_CONTEXT, keys.sign_sk)

def pack_chunk(keys, format_, chunk_addr, chunk_payload):
    if False:
        while True:
            i = 10
    header = struct.pack('<BBBBII', MBOOT_PACK_HEADER_VERSION, format_, 0, 0, chunk_addr, len(chunk_payload))
    chunk = header + chunk_payload
    sig = sign(keys, chunk)
    chunk = chunk + sig
    return chunk

def data_chunks(data, n):
    if False:
        i = 10
        return i + 15
    for i in range(0, len(data), n):
        yield data[i:i + n]

def generate_keys(keys, args):
    if False:
        for i in range(10):
            print('nop')
    keys.generate()
    keys.save()

def pack_dfu(keys, args):
    if False:
        while True:
            i = 10
    chunk_size = int(args.chunk_size[0])
    keys.load()
    (vid_pid, elems) = dfu_read(args.infile[0])
    elems = sorted(elems, key=lambda e: e[0])
    target = []
    full_fw = b''
    full_signature_payload = b''
    for (address, fw) in elems:
        full_fw += fw
        full_signature_payload += struct.pack('<II', address, len(fw))
        for (i, chunk) in enumerate(data_chunks(fw, chunk_size)):
            chunk_addr = address + i * chunk_size
            if args.gzip:
                chunk = compress(chunk)
            chunk = encrypt(keys, chunk)
            chunk = pack_chunk(keys, MBOOT_PACK_CHUNK_FW_GZIP if args.gzip else MBOOT_PACK_CHUNK_FW_RAW, chunk_addr, chunk)
            target.append({'address': chunk_addr, 'data': chunk})
    chunk_addr += chunk_size
    sig = sign(keys, full_fw)
    full_signature_payload += sig
    full_signature_chunk = pack_chunk(keys, MBOOT_PACK_CHUNK_FULL_SIG, chunk_addr, full_signature_payload)
    target.append({'address': chunk_addr, 'data': full_signature_chunk})
    dfu.build(args.outfile[0], [target], vid_pid)
    verify_pack_dfu(keys, args.outfile[0])

def verify_pack_dfu(keys, filename):
    if False:
        while True:
            i = 10
    'Verify packed dfu file against keys. Gathers decrypted binary data.'
    full_sig = pyhy.hydro_sign(MBOOT_PACK_HYDRO_CONTEXT)
    (_, elems) = dfu_read(filename)
    base_addr = None
    binary_data = b''
    for (addr, data) in elems:
        if base_addr is None:
            base_addr = addr
        header = struct.unpack('<BBBBII', data[:12])
        chunk = data[12:12 + header[5]]
        sig = data[12 + header[5]:]
        sig_pass = pyhy.hydro_sign_verify(sig, data[:12] + chunk, MBOOT_PACK_HYDRO_CONTEXT, keys.sign_pk)
        assert sig_pass
        if header[1] == MBOOT_PACK_CHUNK_FULL_SIG:
            actual_sig = chunk[-64:]
        else:
            chunk = pyhy.hydro_secretbox_decrypt(chunk, 0, MBOOT_PACK_HYDRO_CONTEXT, keys.secretbox)
            assert chunk is not None
            if header[1] == MBOOT_PACK_CHUNK_FW_GZIP:
                chunk = zlib.decompress(chunk, wbits=-15)
            full_sig.update(chunk)
            assert addr == base_addr + len(binary_data)
            binary_data += chunk
    full_sig_pass = full_sig.final_verify(actual_sig, keys.sign_pk)
    assert full_sig_pass
    return [{'address': base_addr, 'data': binary_data}]

def unpack_dfu(keys, args):
    if False:
        print('Hello World!')
    keys.load()
    data = verify_pack_dfu(keys, args.infile[0])
    dfu.build(args.outfile[0], [data])

def main():
    if False:
        while True:
            i = 10
    cmd_parser = argparse.ArgumentParser(description='Build signed/encrypted DFU files')
    cmd_parser.add_argument('-k', '--keys', default='mboot_keys.h', help='filename for keys')
    subparsers = cmd_parser.add_subparsers()
    parser_gk = subparsers.add_parser('generate-keys', help='generate keys')
    parser_gk.set_defaults(func=generate_keys)
    parser_ed = subparsers.add_parser('pack-dfu', help='encrypt and sign a DFU file')
    parser_ed.add_argument('-z', '--gzip', action='store_true', help='compress chunks')
    parser_ed.add_argument('chunk_size', nargs=1, help='maximum size in bytes of each chunk')
    parser_ed.add_argument('infile', nargs=1, help='input DFU file')
    parser_ed.add_argument('outfile', nargs=1, help='output DFU file')
    parser_ed.set_defaults(func=pack_dfu)
    parser_dd = subparsers.add_parser('unpack-dfu', help='decrypt a signed/encrypted DFU file')
    parser_dd.add_argument('infile', nargs=1, help='input packed DFU file')
    parser_dd.add_argument('outfile', nargs=1, help='output DFU file')
    parser_dd.set_defaults(func=unpack_dfu)
    args = cmd_parser.parse_args()
    keys = Keys(args.keys)
    args.func(keys, args)
if __name__ == '__main__':
    main()