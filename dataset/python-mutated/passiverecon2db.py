"""Update the database from output of the Zeek script 'passiverecon'"""
import functools
import signal
import sys
from argparse import ArgumentParser
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
from ivre.db import DBPassive, db
from ivre.parser.zeek import ZeekFile
from ivre.passive import getinfos, handle_rec
from ivre.types import Record
from ivre.utils import force_ip2int
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)

def _get_ignore_rules(ignore_spec: Optional[str]) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
    if False:
        print('Hello World!')
    'Executes the ignore_spec file and returns the ignore_rules\n    dictionary.\n\n    '
    ignore_rules: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
    if ignore_spec is not None:
        with open(ignore_spec, 'rb') as fdesc:
            exec(compile(fdesc.read(), ignore_spec, 'exec'), ignore_rules)
    subdict = ignore_rules.get('IGNORENETS')
    if subdict:
        for (subkey, values) in subdict.items():
            subdict[subkey] = [(force_ip2int(val[0]), force_ip2int(val[1])) for val in values]
    return ignore_rules

def rec_iter(zeek_parser: Iterable[Dict[str, Any]], sensor: Optional[str], ignore_rules: Dict[str, Dict[str, List[Tuple[int, int]]]]) -> Generator[Tuple[Optional[int], Record], None, None]:
    if False:
        return 10
    for line in zeek_parser:
        line['timestamp'] = line.pop('ts')
        line['recon_type'] = line['recon_type'][14:]
        yield from handle_rec(sensor, ignore_rules.get('IGNORENETS', {}), ignore_rules.get('NEVERIGNORE', {}), **line)

def main() -> None:
    if False:
        print('Hello World!')
    parser = ArgumentParser(description=__doc__, parents=[db.passive.argparser_insert])
    parser.add_argument('files', nargs='*', metavar='FILE', help='passive_recon log files')
    args = parser.parse_args()
    ignore_rules = _get_ignore_rules(args.ignore_spec)
    if args.test:
        function = DBPassive().insert_or_update_local_bulk
    elif not (args.no_bulk or args.local_bulk) or args.bulk:
        function = db.passive.insert_or_update_bulk
    elif args.local_bulk:
        function = db.passive.insert_or_update_local_bulk
    else:
        function = functools.partial(DBPassive.insert_or_update_bulk, db.passive)
    for fdesc in args.files or [sys.stdin.buffer]:
        function(rec_iter(ZeekFile(fdesc), args.sensor, ignore_rules), getinfos=getinfos)