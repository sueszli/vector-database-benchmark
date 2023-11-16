"""Update the flow database from log files"""
from argparse import ArgumentParser
from ivre import config, utils
from ivre.db import db
from ivre.parser.argus import Argus
from ivre.parser.iptables import Iptables
from ivre.parser.netflow import NetFlow
PARSERS_CHOICE = {'argus': Argus, 'netflow': NetFlow, 'iptables': Iptables}
PARSERS_MAGIC = {b'\x0c\xa5\x01\x00': NetFlow, b'\x83\x10\x00 ': Argus}

def main() -> None:
    if False:
        return 10
    'Update the flow database from log files'
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*', metavar='FILE', help='Files to import in the flow database')
    parser.add_argument('-v', '--verbose', help='verbose mode', action='store_true')
    parser.add_argument('-t', '--type', help='file type', choices=list(PARSERS_CHOICE))
    parser.add_argument('-f', '--pcap-filter', help='pcap filter to apply (when supported)')
    parser.add_argument('-C', '--no-cleanup', help='avoid port cleanup heuristics', action='store_true')
    args = parser.parse_args()
    if args.verbose:
        config.DEBUG = True
    for fname in args.files:
        try:
            fileparser = PARSERS_CHOICE[args.type]
        except KeyError:
            with utils.open_file(fname) as fdesc_tmp:
                try:
                    fileparser = PARSERS_MAGIC[fdesc_tmp.read(4)]
                except KeyError:
                    utils.LOGGER.warning('Cannot find the appropriate parser for file %r', fname)
                    continue
        bulk = db.flow.start_bulk_insert()
        with fileparser(fname, args.pcap_filter) as fdesc:
            for rec in fdesc:
                if not rec:
                    continue
                db.flow.flow2flow(bulk, rec)
        db.flow.bulk_commit(bulk)
    if not args.no_cleanup:
        db.flow.cleanup_flows()