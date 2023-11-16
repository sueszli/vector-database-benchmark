"""Update the flow database from Zeek logs"""
import os
from argparse import ArgumentParser
from typing import Any, Callable, Dict, Iterable
from ivre import config, flow, utils
from ivre.db import db
from ivre.parser.zeek import ZeekFile
from ivre.types import Record
from ivre.types.flow import Bulk

def _zeek2flow(rec: Dict[str, Any]) -> Record:
    if False:
        i = 10
        return i + 15
    'Prepares a document'
    if 'id_orig_h' in rec:
        rec['src'] = rec.pop('id_orig_h')
    elif 'pkt_src' in rec:
        rec['src'] = rec.pop('pkt_src')
    if 'id_resp_h' in rec:
        rec['dst'] = rec.pop('id_resp_h')
    elif 'pkt_dst' in rec:
        rec['dst'] = rec.pop('pkt_dst')
    if 'ts' in rec:
        rec['start_time'] = rec['end_time'] = rec.pop('ts')
    if rec.get('proto', None) == 'icmp':
        (rec['type'], rec['code']) = (rec.pop('id_orig_p'), rec.pop('id_resp_p'))
    elif 'id_orig_p' in rec and 'id_resp_p' in rec:
        (rec['sport'], rec['dport']) = (rec.pop('id_orig_p'), rec.pop('id_resp_p'))
    return rec

def arp2flow(bulk: Bulk, rec: Record) -> None:
    if False:
        for i in range(10):
            print('nop')
    rec['proto'] = 'arp'
    db.flow.any2flow(bulk, 'arp', rec)

def http2flow(bulk: Bulk, rec: Record) -> None:
    if False:
        while True:
            i = 10
    rec['proto'] = 'tcp'
    db.flow.any2flow(bulk, 'http', rec)

def ssh2flow(bulk: Bulk, rec: Record) -> None:
    if False:
        print('Hello World!')
    rec['proto'] = 'tcp'
    db.flow.any2flow(bulk, 'ssh', rec)

def _sip_paths_search_tcp(paths: Iterable[str]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Fills the given proto_dict with transport protocol info found in path\n    '
    for elt in paths:
        info = elt.split(' ')[0].split('/')
        if len(info) != 3:
            continue
        if info[2].lower() in ['tcp', 'tls']:
            return True
    return False

def sip2flow(bulk: Bulk, rec: Record) -> None:
    if False:
        print('Hello World!')
    found_tcp = _sip_paths_search_tcp(rec['response_path']) or _sip_paths_search_tcp(rec['request_path'])
    rec['proto'] = 'tcp' if found_tcp else 'udp'
    db.flow.any2flow(bulk, 'sip', rec)

def snmp2flow(bulk: Bulk, rec: Record) -> None:
    if False:
        while True:
            i = 10
    rec['proto'] = 'udp'
    db.flow.any2flow(bulk, 'snmp', rec)

def ssl2flow(bulk: Bulk, rec: Record) -> None:
    if False:
        print('Hello World!')
    rec['proto'] = 'tcp'
    db.flow.any2flow(bulk, 'ssl', rec)

def rdp2flow(bulk: Bulk, rec: Record) -> None:
    if False:
        i = 10
        return i + 15
    rec['proto'] = 'tcp'
    db.flow.any2flow(bulk, 'rdp', rec)

def dns2flow(bulk: Bulk, rec: Record) -> None:
    if False:
        while True:
            i = 10
    rec['answers'] = [elt.lower() for elt in (rec['answers'] if rec['answers'] else [])]
    rec['query'] = rec['query'].lower() if rec['query'] else None
    db.flow.any2flow(bulk, 'dns', rec)
FUNCTIONS = {'arp': arp2flow, 'conn': db.flow.conn2flow, 'http': http2flow, 'ssh': ssh2flow, 'dns': dns2flow, 'sip': sip2flow, 'snmp': snmp2flow, 'ssl': ssl2flow, 'rdp': rdp2flow}

def any2flow(name: str) -> Callable[[Bulk, Record], None]:
    if False:
        print('Hello World!')

    def inserter(bulk: Bulk, rec: Record) -> None:
        if False:
            return 10
        db.flow.any2flow(bulk, name, rec)
    return inserter

def main() -> None:
    if False:
        print('Hello World!')
    'Update the flow database from Zeek logs'
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('logfiles', nargs='*', metavar='FILE', help='Zeek log files')
    parser.add_argument('-v', '--verbose', help='verbose mode', action='store_true')
    parser.add_argument('-C', '--no-cleanup', help='avoid port cleanup heuristics', action='store_true')
    args = parser.parse_args()
    if args.verbose:
        config.DEBUG = True
    for fname in args.logfiles:
        if not os.path.exists(fname):
            utils.LOGGER.error('File %r does not exist', fname)
            continue
        with ZeekFile(fname) as zeekf:
            bulk = db.flow.start_bulk_insert()
            utils.LOGGER.debug('Parsing %s\n\t%s', fname, 'Fields:\n%s\n' % '\n'.join(('%s: %s' % (f.decode(), t.decode()) for (f, t) in zeekf.field_types)))
            if zeekf.path in FUNCTIONS:
                func = FUNCTIONS[zeekf.path]
            elif zeekf.path in flow.META_DESC:
                func = any2flow(zeekf.path)
            else:
                utils.LOGGER.debug('Log format not (yet) supported for %r', fname)
                continue
            for line in zeekf:
                if not line:
                    continue
                func(bulk, _zeek2flow(line))
            db.flow.bulk_commit(bulk)
            if zeekf.path == 'conn' and (not args.no_cleanup):
                db.flow.cleanup_flows()