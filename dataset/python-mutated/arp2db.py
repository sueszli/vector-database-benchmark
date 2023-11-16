"""Update the flow database from ARP requests in PCAP files"""
import subprocess
from argparse import ArgumentParser
from datetime import datetime
from typing import Iterable, cast
from scapy.all import ARP, Packet, PcapReader
from ivre import config
from ivre.db import db

def reader(fname: str) -> Iterable[Packet]:
    if False:
        return 10
    proc = subprocess.Popen(['tcpdump', '-n', '-r', fname, '-w', '-', 'arp'], stdout=subprocess.PIPE)
    return cast(Iterable[Packet], PcapReader(proc.stdout))

def main() -> None:
    if False:
        i = 10
        return i + 15
    'Update the flow database from ARP requests in PCAP files'
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='*', metavar='FILE', help='PCAP files')
    parser.add_argument('-v', '--verbose', help='verbose mode', action='store_true')
    args = parser.parse_args()
    if args.verbose:
        config.DEBUG = True
    bulk = db.flow.start_bulk_insert()
    for fname in args.files:
        for pkt in reader(fname):
            rec = {'dst': pkt[ARP].pdst, 'src': pkt[ARP].psrc, 'mac_src': pkt[ARP].hwsrc, 'mac_dst': pkt[ARP].hwdst, 'start_time': datetime.fromtimestamp(pkt.time), 'end_time': datetime.fromtimestamp(pkt.time), 'op': pkt.sprintf('%ARP.op%').upper().replace('-', '_'), 'proto': 'arp'}
            if rec['dst'] != '0.0.0.0' and rec['src'] != '0.0.0.0':
                db.flow.any2flow(bulk, 'arp', rec)
    db.flow.bulk_commit(bulk)