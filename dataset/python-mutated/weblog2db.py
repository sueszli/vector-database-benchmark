"""Update the passive database from http server log files"""
from argparse import ArgumentParser
from functools import partial
from sys import stdin
from typing import Dict, Generator, List, Optional, Tuple
from ivre.db import DBPassive, db
from ivre.parser.weblog import WeblogFile
from ivre.passive import getinfos, handle_rec
from ivre.tools.passiverecon2db import _get_ignore_rules
from ivre.types import Record

def rec_iter(filenames: List[str], sensor: Optional[str], ignore_rules: Dict[str, Dict[str, List[Tuple[int, int]]]]) -> Generator[Tuple[Optional[int], Record], None, None]:
    if False:
        for i in range(10):
            print('nop')
    ignorenets = ignore_rules.get('IGNORENETS', {})
    neverignore = ignore_rules.get('NEVERIGNORE', {})
    for fname in filenames:
        with WeblogFile(fname) as fdesc:
            for line in fdesc:
                if not line:
                    continue
                if line.get('useragent') and line['useragent'] != '-':
                    yield from handle_rec(sensor, ignorenets, neverignore, timestamp=line['ts'], uid=None, host=line['host'], srvport=None, recon_type='HTTP_CLIENT_HEADER', source='USER-AGENT', value=line['useragent'], targetval=None)

def main() -> None:
    if False:
        i = 10
        return i + 15
    'Update the flow database from http server log files'
    parser = ArgumentParser(description=__doc__, parents=[db.passive.argparser_insert])
    parser.add_argument('files', nargs='*', metavar='FILE', help='http server log files')
    args = parser.parse_args()
    ignore_rules = _get_ignore_rules(args.ignore_spec)
    if args.test:
        function = DBPassive().insert_or_update_local_bulk
    elif not (args.no_bulk or args.local_bulk) or args.bulk:
        function = db.passive.insert_or_update_bulk
    elif args.local_bulk:
        function = db.passive.insert_or_update_local_bulk
    else:
        function = partial(DBPassive.insert_or_update_bulk, db.passive)
    function(rec_iter(args.files or [stdin.buffer], args.sensor, ignore_rules), getinfos=getinfos)
if __name__ == '__main__':
    main()