from __future__ import annotations
import argparse
import os
from typing import IO
from typing import Sequence

def fix_file(file_obj: IO[bytes]) -> int:
    if False:
        for i in range(10):
            print('nop')
    try:
        file_obj.seek(-1, os.SEEK_END)
    except OSError:
        return 0
    last_character = file_obj.read(1)
    if last_character not in {b'\n', b'\r'} and last_character != b'':
        file_obj.seek(0, os.SEEK_END)
        file_obj.write(b'\n')
        return 1
    while last_character in {b'\n', b'\r'}:
        if file_obj.tell() == 1:
            file_obj.seek(0)
            file_obj.truncate()
            return 1
        file_obj.seek(-2, os.SEEK_CUR)
        last_character = file_obj.read(1)
    position = file_obj.tell()
    remaining = file_obj.read()
    for sequence in (b'\n', b'\r\n', b'\r'):
        if remaining == sequence:
            return 0
        elif remaining.startswith(sequence):
            file_obj.seek(position + len(sequence))
            file_obj.truncate()
            return 1
    return 0

def main(argv: Sequence[str] | None=None) -> int:
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*', help='Filenames to fix')
    args = parser.parse_args(argv)
    retv = 0
    for filename in args.filenames:
        with open(filename, 'rb+') as file_obj:
            ret_for_file = fix_file(file_obj)
            if ret_for_file:
                print(f'Fixing {filename}')
            retv |= ret_for_file
    return retv
if __name__ == '__main__':
    raise SystemExit(main())