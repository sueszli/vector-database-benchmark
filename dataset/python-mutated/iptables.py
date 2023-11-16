"""Support for Iptables log from syslog files."""
import datetime
from typing import Any, Dict, Optional
from ivre.parser import Parser
from ivre.utils import LOGGER

class Iptables(Parser):
    """Iptables log generator from a syslog file descriptor."""

    def __init__(self, fname: str, pcap_filter: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        'Init Ipatbles class.'
        if pcap_filter is not None:
            LOGGER.warning('PCAP filter not supported in Iptables')
        super().__init__(fname)

    def parse_line(self, line: bytes) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Process current line in Parser.__next__.'
        field_idx = line.find(b'IN=')
        if field_idx < 0:
            return next(self)
        fields: Dict[str, Any] = {key.decode().lower(): value.decode() for (key, value) in (val.split(b'=', 1) if b'=' in val else (val, b'') for val in line[field_idx:].rstrip(b'\r\n').split())}
        try:
            fields['start_time'] = datetime.datetime.strptime(line[:15].decode(), '%b %d %H:%M:%S')
        except ValueError:
            return next(self)
        fields['proto'] = fields['proto'].lower()
        if fields['proto'] in ('udp', 'tcp'):
            fields['sport'] = int(fields.pop('spt'))
            fields['dport'] = int(fields.pop('dpt'))
        fields['cspkts'] = fields['scpkts'] = 0
        fields['scbytes'] = fields['csbytes'] = 0
        fields['end_time'] = fields['start_time']
        return fields