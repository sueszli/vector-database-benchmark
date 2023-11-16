"""This tool can be used to query manufacturers for MAC addresses.

"""
import json
from argparse import ArgumentParser
from sys import stdout
from ivre import utils

def main() -> None:
    if False:
        i = 10
        return i + 15
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--json', '-j', action='store_true', help='Output JSON data.')
    parser.add_argument('mac', nargs='*', metavar='MAC', help='Display manufacturers for specified MAC addresses.')
    args = parser.parse_args()
    for addr in args.mac:
        info = utils.mac2manuf(addr)
        if args.json:
            res = {'addr': addr}
            if info:
                if isinstance(info, tuple):
                    if info[0]:
                        res['manufacturer_code'] = info[0]
                    if info[1:] and info[1]:
                        res['manufacturer_name'] = info[1]
                else:
                    res['manufacturer_name'] = info
            json.dump(res, stdout)
            print()
        elif isinstance(info, tuple):
            print(f"{addr} {' / '.join((i for i in info if i))}")
        else:
            print(f'{addr} {info}')