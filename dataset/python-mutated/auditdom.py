"""Audit a DNS domain to produce an XML or JSON result similar to an
Nmap script result."""
import argparse
import json
import sys
from datetime import datetime
from shlex import quote
from typing import Any, Iterable, Optional
from ivre import VERSION
from ivre.activecli import displayfunction_nmapxml
from ivre.analyzer.dns import AXFRChecker, DNSSRVChecker, TLSRPTChecker
from ivre.types import Record
from ivre.utils import LOGGER, serialize

def main() -> None:
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Output as JSON rather than XML.')
    parser.add_argument('--ipv4', '-4', action='store_true', help='Use only IPv4.')
    parser.add_argument('--ipv6', '-6', action='store_true', help='Use only IPv6.')
    parser.add_argument('domains', metavar='DOMAIN', nargs='+', help='domains to check')
    args = parser.parse_args()
    if args.json:

        def displayfunction(cur: Iterable[Record], scan: Optional[Any]=None) -> None:
            if False:
                i = 10
                return i + 15
            if scan is not None:
                LOGGER.debug('Scan not displayed in JSON mode')
            for rec in cur:
                print(json.dumps(rec, default=serialize))
    else:
        displayfunction = displayfunction_nmapxml
    start = datetime.now()
    scan = {'scanner': 'ivre auditdom', 'start': start.strftime('%s'), 'startstr': str(start), 'version': VERSION, 'xmloutputversion': '1.04', 'args': ' '.join(sys.argv[:1] + [quote(arg) for arg in sys.argv[1:]]), 'scaninfos': [{'type': 'audit DNS domain', 'protocol': 'dig', 'numservices': 1, 'services': '53'}]}
    results = [rec for domain in args.domains for test in [AXFRChecker, DNSSRVChecker, TLSRPTChecker] for rec in test(domain).test(v4=not args.ipv6, v6=not args.ipv4)]
    end = datetime.now()
    scan['end'] = end.strftime('%s')
    scan['endstr'] = str(end)
    scan['elapsed'] = str((end - start).total_seconds())
    displayfunction(results, scan=scan)