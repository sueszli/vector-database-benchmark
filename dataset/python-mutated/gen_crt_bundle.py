from __future__ import with_statement
import argparse
import csv
import os
import re
import struct
import sys
import textwrap
from io import open
try:
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
except ImportError:
    print('The cryptography package is not installed.Please refer to the Get Started section of the ESP-IDF Programming Guide for setting up the required packages.')
    raise
ca_bundle_bin_file = 'x509_crt_bundle'
quiet = False

def status(msg):
    if False:
        i = 10
        return i + 15
    'Print status message to stderr'
    if not quiet:
        critical(msg)

def critical(msg):
    if False:
        i = 10
        return i + 15
    'Print critical message to stderr'
    sys.stderr.write('gen_crt_bundle.py: ')
    sys.stderr.write(msg)
    sys.stderr.write('\n')

class CertificateBundle:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.certificates = []
        self.compressed_crts = []
        if os.path.isfile(ca_bundle_bin_file):
            os.remove(ca_bundle_bin_file)

    def add_from_path(self, crts_path):
        if False:
            i = 10
            return i + 15
        found = False
        for file_path in os.listdir(crts_path):
            found |= self.add_from_file(os.path.join(crts_path, file_path))
        if found is False:
            raise InputError('No valid x509 certificates found in %s' % crts_path)

    def add_from_file(self, file_path):
        if False:
            return 10
        try:
            if file_path.endswith('.pem'):
                status('Parsing certificates from %s' % file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    crt_str = f.read()
                    self.add_from_pem(crt_str)
                    return True
            elif file_path.endswith('.der'):
                status('Parsing certificates from %s' % file_path)
                with open(file_path, 'rb') as f:
                    crt_str = f.read()
                    self.add_from_der(crt_str)
                    return True
        except ValueError:
            critical('Invalid certificate in %s' % file_path)
            raise InputError('Invalid certificate')
        return False

    def add_from_pem(self, crt_str):
        if False:
            i = 10
            return i + 15
        'A single PEM file may have multiple certificates'
        crt = ''
        count = 0
        start = False
        for strg in crt_str.splitlines(True):
            if strg == '-----BEGIN CERTIFICATE-----\n' and start is False:
                crt = ''
                start = True
            elif strg == '-----END CERTIFICATE-----\n' and start is True:
                crt += strg + '\n'
                start = False
                self.certificates.append(x509.load_pem_x509_certificate(crt.encode(), default_backend()))
                count += 1
            if start is True:
                crt += strg
        if count == 0:
            raise InputError('No certificate found')
        status('Successfully added %d certificates' % count)

    def add_from_der(self, crt_str):
        if False:
            for i in range(10):
                print('nop')
        self.certificates.append(x509.load_der_x509_certificate(crt_str, default_backend()))
        status('Successfully added 1 certificate')

    def create_bundle(self):
        if False:
            return 10
        self.certificates = sorted(self.certificates, key=lambda cert: cert.subject.public_bytes(default_backend()))
        bundle = struct.pack('>H', len(self.certificates))
        for crt in self.certificates:
            'Read the public key as DER format'
            pub_key = crt.public_key()
            pub_key_der = pub_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
            ' Read the subject name as DER format '
            sub_name_der = crt.subject.public_bytes(default_backend())
            name_len = len(sub_name_der)
            key_len = len(pub_key_der)
            len_data = struct.pack('>HH', name_len, key_len)
            bundle += len_data
            bundle += sub_name_der
            bundle += pub_key_der
        return bundle

    def add_with_filter(self, crts_path, filter_path):
        if False:
            i = 10
            return i + 15
        filter_set = set()
        with open(filter_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f, delimiter=',')
            next(csv_reader)
            for row in csv_reader:
                filter_set.add(row[1])
        status('Parsing certificates from %s' % crts_path)
        crt_str = []
        with open(crts_path, 'r', encoding='utf-8') as f:
            crt_str = f.read()
            pem_crts = re.findall('(^.+?)\\n(=+\\n[\\s\\S]+?END CERTIFICATE-----\\n)', crt_str, re.MULTILINE)
            filtered_crts = ''
            for (name, crt) in pem_crts:
                if name in filter_set:
                    filtered_crts += crt
        self.add_from_pem(filtered_crts)

class InputError(RuntimeError):

    def __init__(self, e):
        if False:
            i = 10
            return i + 15
        super(InputError, self).__init__(e)

def main():
    if False:
        print('Hello World!')
    global quiet
    parser = argparse.ArgumentParser(description='ESP-IDF x509 certificate bundle utility')
    parser.add_argument('--quiet', '-q', help="Don't print non-critical status messages to stderr", action='store_true')
    parser.add_argument('--input', '-i', nargs='+', required=True, help='Paths to the custom certificate folders or files to parse, parses all .pem or .der files')
    parser.add_argument('--filter', '-f', help='Path to CSV-file where the second columns contains the name of the certificates                         that should be included from cacrt_all.pem')
    parser.add_argument('--asm', '-S', action='store_true', default=False, help='Output an asm file for use with gas, rather than a binary file')
    parser.add_argument('--symbol', help='The symbol to define', default='x509_crt_bundle')
    parser.add_argument('--output', '-o', help='The output file', default=None)
    args = parser.parse_args()
    quiet = args.quiet
    bundle = CertificateBundle()
    for path in args.input:
        if os.path.isfile(path):
            if os.path.basename(path) == 'cacrt_all.pem' and args.filter:
                bundle.add_with_filter(path, args.filter)
            else:
                bundle.add_from_file(path)
        elif os.path.isdir(path):
            bundle.add_from_path(path)
        else:
            raise InputError('Invalid --input=%s, is neither file nor folder' % args.input)
    status('Successfully added %d certificates in total' % len(bundle.certificates))
    crt_bundle = bundle.create_bundle()
    if args.asm:
        symbol = args.symbol
        filename = args.output or ca_bundle_bin_file + '.S'
        with open(filename, 'w') as f:
            print(textwrap.dedent(f"                // Generated from {' '.join(args.input)} with {len(bundle.certificates)} certificates\n                .data\n                .section .rodata.embedded\n\n                .global {symbol}\n                .global _binary_{symbol}_start\n                .global _binary_{symbol}_end\n                {symbol}:\n                _binary_{symbol}_start:\n                "), file=f)
            for i in range(0, len(crt_bundle), 16):
                chunk = crt_bundle[i:i + 16]
                formatted = ', '.join((f'0x{byte:02x}' for byte in chunk))
                print(f'.byte {formatted}', file=f)
            print(textwrap.dedent(f'                _binary_{symbol}_end:\n\n                {symbol}_length:\n                .word {len(crt_bundle)}\n            '), file=f)
    else:
        filename = args.output or ca_bundle_bin_file
        with open(filename, 'wb') as f:
            f.write(crt_bundle)
if __name__ == '__main__':
    try:
        main()
    except InputError as e:
        print(e)
        sys.exit(2)