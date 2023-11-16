"""Generate scan results (XML or JSON) from local commands such as
netstat or ss

"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from typing import Dict, Generator, Set, Tuple
from ivre import utils
from ivre.activecli import displayfunction_nmapxml
from ivre.types import DBCursor
from ivre.types.active import NmapHost

class LocalPorts:
    cmd_addrs = ['ip', 'addr', 'show']
    cmd_openports = ['ss', '-ltunp']
    match_addrs = re.compile('^ +inet (?P<addr>[0-9\\.]+)/')
    match_openports = re.compile('^(?P<proto>tcp|udp) +(?P<state>[A-Z-]+) +[0-9]+ +[0-9]+ +(?P<addr>[0-9\\.]+):(?P<port>[0-9]+) +' + re.escape('0.0.0.0:*') + ' +(users:\\(\\("(?P<progname>[^"]*)",pid=[0-9]+,fd=[0-9]+\\)\\))?')

    def get_addrs(self) -> Generator[str, None, None]:
        if False:
            i = 10
            return i + 15
        with subprocess.Popen(self.cmd_addrs, stdout=subprocess.PIPE) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                m = self.match_addrs.search(line.decode())
                if m is None:
                    continue
                yield m.groupdict()['addr']

    def parse(self) -> Dict[str, Dict[Tuple[str, int], Set[str]]]:
        if False:
            while True:
                i = 10
        addresses = set(self.get_addrs())
        results: Dict[str, Dict[Tuple[str, int], Set[str]]] = {}
        with subprocess.Popen(self.cmd_openports, stdout=subprocess.PIPE) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                m = self.match_openports.search(line.decode())
                if m is None:
                    continue
                md = m.groupdict()
                if {'tcp': 'LISTEN', 'udp': 'UNCONN'}[md['proto']] != md['state']:
                    utils.LOGGER.warning('Weird state %(state)s (proto %(proto)s)', md)
                for addr in addresses if md['addr'] == '0.0.0.0' else [md['addr']]:
                    progs = results.setdefault(addr, {}).setdefault((md['proto'], int(md['port'])), set())
                    if md['progname'] is not None:
                        progs.add(md['progname'])
            return results

    def get_scan_results(self) -> Generator[NmapHost, None, None]:
        if False:
            i = 10
            return i + 15
        starttime = datetime.now()
        results = sorted(self.parse().items(), key=lambda a_p: utils.ip2int(a_p[0]))
        endtime = datetime.now()
        for (addr, ports) in results:
            rec = {'addr': addr, 'starttime': starttime, 'endtime': endtime}
            for ((proto, portnum), progs) in sorted(ports.items()):
                port = {'protocol': proto, 'port': portnum}
                if progs:
                    port['scripts'] = [{'id': 'localscan', 'output': 'Program%s: %s' % ('s' if len(progs) > 1 else '', ', '.join(sorted(progs))), 'localscan': {'programs': sorted(progs)}}]
                rec.setdefault('ports', []).append(port)
            yield rec

def main() -> None:
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Output as JSON rather than XML.')
    args = parser.parse_args()
    if args.json:

        def displayfunction(cur: DBCursor) -> None:
            if False:
                print('Hello World!')
            for rec in cur:
                json.dump(rec, sys.stdout, default=utils.serialize)
                sys.stdout.write('\n')
    else:
        displayfunction = displayfunction_nmapxml
    displayfunction(LocalPorts().get_scan_results())