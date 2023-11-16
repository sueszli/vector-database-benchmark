"""Parse NMAP scan results and add them in DB."""
import os
import sys
from argparse import ArgumentParser
from typing import Generator, Iterable, List
import ivre.db
import ivre.utils
import ivre.xmlnmap
from ivre.tags.active import set_auto_tags, set_openports_attribute
from ivre.types import Record
from ivre.view import nmap_record_to_view

def recursive_filelisting(base_directories: Iterable[str], error: List[bool]) -> Generator[str, None, None]:
    if False:
        while True:
            i = 10
    'Iterator on filenames in base_directories. Ugly hack: error is a\n    one-element list that will be set to True if one of the directories in\n    base_directories does not exist.\n\n    '
    for base_directory in base_directories:
        if not os.path.exists(base_directory):
            ivre.utils.LOGGER.warning('directory %r does not exist', base_directory)
            error[0] = True
            continue
        if not os.path.isdir(base_directory):
            yield base_directory
            continue
        for (root, _, files) in os.walk(base_directory):
            for leaffile in files:
                yield os.path.join(root, leaffile)

def main() -> None:
    if False:
        i = 10
        return i + 15
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('scan', nargs='*', metavar='SCAN', help='Scan results')
    parser.add_argument('-c', '--categories', default='', help='Scan categories.')
    parser.add_argument('-s', '--source', default=None, help='Scan source.')
    parser.add_argument('-t', '--test', action='store_true', help='Test mode (JSON output).')
    parser.add_argument('--tags', metavar='TAG:LEVEL:INFO[,TAG:LEVEL:INFO]', help='Add tags to the results; e.g. --tags=CDN:info:"My CDN",Honeypot:warning:"My Masscanned Honeypot"')
    parser.add_argument('--test-normal', action='store_true', help='Test mode ("normal" Nmap output).')
    parser.add_argument('--ports', '--port', action='store_true', help='Store only hosts with a "ports" element.')
    parser.add_argument('--open-ports', action='store_true', help='Store only hosts with open ports.')
    parser.add_argument('--masscan-probes', nargs='+', metavar='PROBE', help='Additional Nmap probes to use when trying to match Masscan results against Nmap service fingerprints.')
    parser.add_argument('--zgrab-port', metavar='PORT', help='Port used for the zgrab scan. This might be needed since the port number does not appear in theresult.')
    parser.add_argument('-r', '--recursive', action='store_true', help='Import all files from given directories.')
    parser.add_argument('--update-view', action='store_true', help='Merge hosts in current view')
    parser.add_argument('--no-update-view', action='store_true', help='Do not merge hosts in current view (default)')
    args = parser.parse_args()
    database = ivre.db.db.nmap
    categories = args.categories.split(',') if args.categories else []
    tags = [{'value': value, 'type': type_, 'info': [info]} if info else {'value': value, 'type': type_} for (value, type_, info) in (tag.split(':', 3) for tag in (args.tags.split(',') if args.tags else []))]
    if args.test:
        args.update_view = False
        args.no_update_view = True
        database = ivre.db.DBNmap()
    if args.test_normal:
        args.update_view = False
        args.no_update_view = True
        database = ivre.db.DBNmap(output_mode='normal')
    error = [False]
    if args.recursive:
        scans = recursive_filelisting(args.scan, error)
    else:
        scans = args.scan
    if not args.update_view or args.no_update_view:
        callback = None
    else:

        def callback(x: Record) -> None:
            if False:
                while True:
                    i = 10
            result = nmap_record_to_view(x)
            set_auto_tags(result, update_openports=False)
            set_openports_attribute(result)
            result['infos'] = {}
            for func in [ivre.db.db.data.country_byip, ivre.db.db.data.as_byip, ivre.db.db.data.location_byip]:
                result['infos'].update(func(result['addr']) or {})
            ivre.db.db.view.store_or_merge_host(result)
        ivre.db.db.view.start_store_hosts()
    count = 0
    for scan in scans:
        if not os.path.exists(scan):
            ivre.utils.LOGGER.warning('file %r does not exist', scan)
            error[0] = True
            continue
        try:
            if database.store_scan(scan, categories=categories, source=args.source, tags=tags, needports=args.ports, needopenports=args.open_ports, masscan_probes=args.masscan_probes, callback=callback, zgrab_port=args.zgrab_port):
                count += 1
        except Exception:
            ivre.utils.LOGGER.warning('Exception (file %r)', scan, exc_info=True)
            error[0] = True
    if callback is not None:
        ivre.db.db.view.stop_store_hosts()
    ivre.utils.LOGGER.info('%d results imported.', count)
    sys.exit(error[0])