"""Support for Argus log files"""
import datetime
from typing import Any, BinaryIO, Dict, Optional, Union
from ivre.parser import CmdParser

class Argus(CmdParser):
    """Argus log generator"""
    fields = ['proto', 'dir', 'saddr', 'sport', 'daddr', 'dport', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'stime', 'ltime']
    aggregation = ['saddr', 'sport', 'daddr', 'dport', 'proto']
    timefmt = '%s.%f'

    def __init__(self, fdesc: Union[str, BinaryIO], pcap_filter: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        'Creates the Argus object.\n\n        fdesc: a file-like object or a filename\n        pcap_filter: a PCAP filter to use with racluster\n        '
        cmd = ['racluster', '-u', '-n', '-c', ',', '-m']
        cmd.extend(self.aggregation)
        cmd.append('-s')
        cmd.extend(self.fields)
        cmd.extend(['-r', fdesc if isinstance(fdesc, str) else '-'])
        if pcap_filter is not None:
            cmd.extend(['--', pcap_filter])
        super().__init__(cmd, {} if isinstance(fdesc, str) else {'stdin': fdesc})
        self.fdesc.readline()

    @classmethod
    def parse_line(cls, line: bytes) -> Dict[str, Any]:
        if False:
            return 10
        fields: Dict[str, Any] = {name: val.strip().decode() for (name, val) in zip(cls.fields, line.split(b','))}
        for fld in ['sport', 'dport']:
            try:
                fields[fld] = int(fields[fld], 16 if fields[fld].startswith('0x') else 10)
            except ValueError:
                if not fields[fld]:
                    del fields[fld]
        fields['src'] = fields.pop('saddr')
        fields['dst'] = fields.pop('daddr')
        fields['csbytes'] = int(fields.pop('sbytes'))
        fields['cspkts'] = int(fields.pop('spkts'))
        fields['scbytes'] = int(fields.pop('dbytes'))
        fields['scpkts'] = int(fields.pop('dpkts'))
        fields['start_time'] = datetime.datetime.fromtimestamp(float(fields.pop('stime')))
        fields['end_time'] = datetime.datetime.fromtimestamp(float(fields.pop('ltime')))
        return fields