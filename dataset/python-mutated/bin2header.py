import argparse
import sys
from pathlib import Path

def bin2header(comment, data, var_name, extern=False):
    if False:
        return 10
    yield comment
    yield '#include <cstddef>'
    if extern:
        yield f'extern const char {var_name}[];'
        yield f'extern const std::size_t {var_name}_len;'
    yield f'const char {var_name}[] = {{'
    indent = '  '
    for i in range(0, len(data), 12):
        hex_chunk = ', '.join((f'0x{x:02x}' for x in data[i:][:12]))
        yield (indent + hex_chunk + ',')
    yield (indent + '0x00 // Terminating null byte')
    yield '};'
    yield f'const std::size_t {var_name}_len = {len(data)};'

def main():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='Generate binary header output')
    parser.add_argument('-i', '--input', required=True, help='Input file', type=Path)
    parser.add_argument('-o', '--out', required=True, help='Output file', type=Path)
    parser.add_argument('-v', '--var', required=True, help='Variable name to use in file')
    parser.add_argument('-e', '--extern', action='store_true', help="Add 'extern' declaration")
    args = parser.parse_args()
    argv_pretty = ' '.join((Path(arg).name if '/' in arg or '\\' in arg else arg for arg in sys.argv))
    comment = f'/* This file was generated using {argv_pretty} */'
    out = bin2header(comment, args.input.read_bytes(), args.var, args.extern)
    args.out.write_text('\n'.join(out))
if __name__ == '__main__':
    sys.exit(main())