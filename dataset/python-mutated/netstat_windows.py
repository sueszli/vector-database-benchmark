"""
    jc - JSON Convert Windows `netstat` command output parser
"""
from typing import Dict, List
POSSIBLE_PROTOCOLS = ('TCP', 'UDP', 'TCPv6', 'UDPv6')

def normalize_headers(headers: str):
    if False:
        print('Hello World!')
    '\n        Normalizes the headers to match the jc netstat parser style\n        (local_address -> local_address, local_port...).\n    '
    headers = headers.lower().strip()
    headers = headers.replace('local address', 'local_address')
    headers = headers.replace('foreign address', 'foreign_address')
    return headers.split()

def parse(cleandata: List[str]):
    if False:
        return 10
    '\n    Main text parsing function for Windows netstat\n\n    Parameters:\n\n        cleandata:   (string)  text data to parse\n\n    Returns:\n\n        List of Dictionaries. Raw structured data.\n    '
    raw_output = []
    cleandata.pop(0)
    headers = normalize_headers(cleandata.pop(0))
    for line in cleandata:
        line = line.strip()
        if not line.startswith(POSSIBLE_PROTOCOLS):
            line_data = raw_output.pop(len(raw_output) - 1)
            line_data['program_name'] = line
            raw_output.append(line_data)
            continue
        line_data = line.split()
        line_data: Dict[str, str] = dict(zip(headers, line_data))
        for key in list(line_data.keys()):
            if key == 'local_address':
                (local_address, local_port) = line_data[key].rsplit(':', maxsplit=1)
                line_data['local_address'] = local_address
                line_data['local_port'] = local_port
                continue
            if key == 'foreign_address':
                (foreign_address, foreign_port) = line_data[key].rsplit(':', maxsplit=1)
                line_data['foreign_address'] = foreign_address
                line_data['foreign_port'] = foreign_port
                continue
            if key == 'proto' and 'state' in headers and (line_data['proto'] == 'UDP'):
                next_header = headers.index('state') + 1
                if len(headers) > next_header:
                    next_header = headers[next_header]
                    line_data[next_header] = line_data['state']
                    line_data['state'] = ''
        raw_output.append(line_data)
    return raw_output