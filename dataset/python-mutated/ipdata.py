"""This tool can be used to manage IP addresses related data, such as
AS number and country information.

"""
import json
from argparse import ArgumentParser
from sys import stdout
from typing import Any, Callable, Dict, List, Tuple, cast
from ivre import geoiputils, utils
from ivre.db import DBData, db
from ivre.tags import add_tags, gen_addr_tags

def main() -> None:
    if False:
        while True:
            i = 10
    parser = ArgumentParser(description=__doc__)
    torun: List[Tuple[Callable, list, dict]] = []
    parser.add_argument('--download', action='store_true', help='Fetch all data files.')
    parser.add_argument('--import-all', action='store_true', help='Create all CSV files for reverse lookups.')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode.')
    parser.add_argument('--json', '-j', action='store_true', help='Output JSON data.')
    parser.add_argument('ip', nargs='*', metavar='IP', help='Display results for specified IP addresses.')
    parser.add_argument('--from-db', metavar='DB_URL', help="Get data from the provided URL instead of using IVRE's configuration.")
    args = parser.parse_args()
    if args.from_db:
        dbase = DBData.from_url(args.from_db)
        dbase.globaldb = db
    else:
        dbase = db.data
    if args.download:
        geoiputils.download_all(verbose=not args.quiet)
        dbase.reload_files()
    if args.import_all:
        torun.append((cast(Callable, dbase.build_dumps), [], {}))
    for (function, fargs, fkargs) in torun:
        function(*fargs, **fkargs)
    for addr in args.ip:
        if addr.isdigit():
            addr = utils.int2ip(int(addr))
        if args.json:
            res = {'addr': addr}
            res.update(dbase.infos_byip(addr) or {})
            json.dump(res, stdout)
            print()
        else:
            print(addr)
            for subinfo in [dbase.as_byip(addr), dbase.location_byip(addr)]:
                for (key, value) in (subinfo or {}).items():
                    print('    %s %s' % (key, value))
            info: Dict[str, Any] = {}
            add_tags(info, gen_addr_tags(addr))
            for tag in info.get('tags', []):
                if tag.get('info'):
                    print(f"    {tag['value']}: {', '.join(tag['info'])}")
                else:
                    print(f"    {tag['value']}")