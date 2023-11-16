"""jc - JSON Convert `ping` command output streaming parser

> This streaming parser outputs JSON Lines (cli) or returns an Iterable of
> Dictionaries (module)

Supports `ping` and `ping6` output.

Usage (cli):

    $ ping 1.2.3.4 | jc --ping-s

> Note: When piping `jc` converted `ping` output to other processes it may
> appear the output is hanging due to the OS pipe buffers. This is because
> `ping` output is too small to quickly fill up the buffer. Use the `-u`
> option to unbuffer the `jc` output if you would like immediate output.
> See the [readme](https://github.com/kellyjonbrazil/jc/tree/master#unbuffering-output)
> for more information.

Usage (module):

    import jc

    result = jc.parse('ping_s', ping_command_output.splitlines())
    for item in result:
        # do something

Schema:

    {
      "type":                       string,   # [0]
      "source_ip":                  string,
      "destination_ip":             string,
      "sent_bytes":                 integer,
      "pattern":                    string,   # null if not set
      "destination":                string,
      "timestamp":                  float,
      "response_bytes":             integer,
      "response_ip":                string,
      "icmp_seq":                   integer,
      "ttl":                        integer,
      "time_ms":                    float,
      "duplicate":                  boolean,
      "packets_transmitted":        integer,
      "packets_received":           integer,
      "packet_loss_percent":        float,
      "duplicates":                 integer,
      "errors":                     integer,  # null if not set
      "corrupted":                  integer,  # null if not set
      "round_trip_ms_min":          float,    # null if not set
      "round_trip_ms_avg":          float,    # null if not set
      "round_trip_ms_max":          float,    # null if not set
      "round_trip_ms_stddev":       float,    # null if not set

      # below object only exists if using -qq or ignore_exceptions=True
      "_jc_meta": {
        "success":                  boolean,  # false if error parsing
        "error":                    string,   # exists if "success" is false
        "line":                     string    # exists if "success" is false
      }
    }

    [0] 'reply', 'timeout', 'summary', etc. See `_error_type.type_map`
        for all options.

Examples:

    $ ping 1.1.1.1 | jc --ping-s
    {"type":"reply","destination_ip":"1.1.1.1","sent_bytes":56,"patte...}
    {"type":"reply","destination_ip":"1.1.1.1","sent_bytes":56,"patte...}
    {"type":"reply","destination_ip":"1.1.1.1","sent_bytes":56,"patte...}
    ...

    $ ping 1.1.1.1 | jc --ping-s -r
    {"type":"reply","destination_ip":"1.1.1.1","sent_bytes":"56","patte...}
    {"type":"reply","destination_ip":"1.1.1.1","sent_bytes":"56","patte...}
    {"type":"reply","destination_ip":"1.1.1.1","sent_bytes":"56","patte...}
    ...
"""
import re
import string
import ipaddress
import jc.utils
from jc.streaming import add_jc_meta, streaming_input_type_check, streaming_line_input_type_check, raise_or_yield
from jc.exceptions import ParseError

class info:
    """Provides parser metadata (version, author, etc.)"""
    version = '1.4'
    description = '`ping` and `ping6` command streaming parser'
    author = 'Kelly Brazil'
    author_email = 'kellyjonbrazil@gmail.com'
    compatible = ['linux', 'darwin', 'freebsd']
    tags = ['command']
    streaming = True
__version__ = info.version

def _process(proc_data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Final processing to conform to the schema.\n\n    Parameters:\n\n        proc_data:   (Dictionary) raw structured data to process\n\n    Returns:\n\n        Dictionary. Structured data to conform to the schema.\n    '
    int_list = {'sent_bytes', 'packets_transmitted', 'packets_received', 'response_bytes', 'icmp_seq', 'ttl', 'duplicates', 'vr', 'hl', 'tos', 'len', 'id', 'flg', 'off', 'pro', 'cks', 'errors', 'corrupted'}
    float_list = {'packet_loss_percent', 'round_trip_ms_min', 'round_trip_ms_avg', 'round_trip_ms_max', 'round_trip_ms_stddev', 'timestamp', 'time_ms'}
    for key in proc_data:
        if key in int_list:
            proc_data[key] = jc.utils.convert_to_int(proc_data[key])
        if key in float_list:
            proc_data[key] = jc.utils.convert_to_float(proc_data[key])
    return proc_data

class _state:
    os_detected = None
    linux = None
    bsd = None
    ipv4 = None
    hostname = None
    destination_ip = None
    sent_bytes = None
    pattern = None
    footer = False
    packets_transmitted = None
    packets_received = None
    packet_loss_percent = None
    time_ms = None
    duplicates = None
    corrupted = None
    errors = None
    round_trip_ms_min = None
    round_trip_ms_avg = None
    round_trip_ms_max = None
    round_trip_ms_stddev = None

def _ipv6_in(line):
    if False:
        return 10
    line_list = line.replace('(', ' ').replace(')', ' ').replace(',', ' ').replace('%', ' ').split()
    ipv6 = False
    for item in line_list:
        try:
            _ = ipaddress.IPv6Address(item)
            ipv6 = True
        except Exception:
            pass
    return ipv6

def _error_type(line):
    if False:
        for i in range(10):
            print('nop')
    type_map = {'Destination Net Unreachable': 'destination_net_unreachable', 'Destination Host Unreachable': 'destination_host_unreachable', 'Destination Protocol Unreachable': 'destination_protocol_unreachable', 'Destination Port Unreachable': 'destination_port_unreachable', 'Frag needed and DF set': 'frag_needed_and_df_set', 'Source Route Failed': 'source_route_failed', 'Destination Net Unknown': 'destination_net_unknown', 'Destination Host Unknown': 'destination_host_unknown', 'Source Host Isolated': 'source_host_isolated', 'Destination Net Prohibited': 'destination_net_prohibited', 'Destination Host Prohibited': 'destination_host_prohibited', 'Destination Net Unreachable for Type of Service': 'destination_net_unreachable_for_type_of_service', 'Destination Host Unreachable for Type of Service': 'destination_host_unreachable_for_type_of_service', 'Packet filtered': 'packet_filtered', 'Precedence Violation': 'precedence_violation', 'Precedence Cutoff': 'precedence_cutoff', 'Dest Unreachable, Bad Code': 'dest_unreachable_bad_code', 'Redirect Network': 'redirect_network', 'Redirect Host': 'redirect_host', 'Redirect Type of Service and Network': 'redirect_type_of_service_and_network', 'Redirect, Bad Code': 'redirect_bad_code', 'Time to live exceeded': 'time_to_live_exceeded', 'Frag reassembly time exceeded': 'frag_reassembly_time_exceeded', 'Time exceeded, Bad Code': 'time_exceeded_bad_code'}
    for (err_type, code) in type_map.items():
        if err_type in line:
            return code
    return None

def _bsd_parse(line, s):
    if False:
        print('Hello World!')
    output_line = {}
    if line.startswith('PING '):
        s.destination_ip = line.split()[2].lstrip('(').rstrip(':').rstrip(')')
        s.sent_bytes = line.split()[3]
        return None
    if line.startswith('PING6('):
        line = line.replace('(', ' ').replace(')', ' ').replace('=', ' ')
        s.source_ip = line.split()[4]
        s.destination_ip = line.split()[6]
        s.sent_bytes = line.split()[1]
        return None
    if line.startswith('---'):
        s.footer = True
        return None
    if s.footer:
        if 'packets transmitted' in line:
            if ' duplicates,' in line:
                s.packets_transmitted = line.split()[0]
                s.packets_received = line.split()[3]
                s.packet_loss_percent = line.split()[8].rstrip('%')
                s.duplicates = line.split()[6].lstrip('+')
                return None
            s.packets_transmitted = line.split()[0]
            s.packets_received = line.split()[3]
            s.packet_loss_percent = line.split()[6].rstrip('%')
            s.duplicates = '0'
            return None
        split_line = line.split(' = ')[1]
        split_line = split_line.split('/')
        output_line = {'type': 'summary', 'destination_ip': s.destination_ip or None, 'sent_bytes': s.sent_bytes or None, 'pattern': s.pattern or None, 'packets_transmitted': s.packets_transmitted or None, 'packets_received': s.packets_received or None, 'packet_loss_percent': s.packet_loss_percent or None, 'duplicates': s.duplicates or None, 'round_trip_ms_min': split_line[0], 'round_trip_ms_avg': split_line[1], 'round_trip_ms_max': split_line[2], 'round_trip_ms_stddev': split_line[3].replace(' ms', '')}
        return output_line
    if not _ipv6_in(line):
        if line.startswith('Request timeout for '):
            output_line = {'type': 'timeout', 'destination_ip': s.destination_ip or None, 'sent_bytes': s.sent_bytes or None, 'pattern': s.pattern or None, 'icmp_seq': line.split()[4]}
            return output_line
        err = _error_type(line)
        if err:
            output_line = {'type': err}
            try:
                output_line['bytes'] = line.split()[0]
                output_line['destination_ip'] = s.destination_ip
                output_line['response_ip'] = line.split()[4].strip(':').strip('(').strip(')')
            except Exception:
                pass
            return output_line
        if ' bytes from ' in line:
            line = line.replace(':', ' ').replace('=', ' ')
            output_line = {'type': 'reply', 'destination_ip': s.destination_ip or None, 'sent_bytes': s.sent_bytes or None, 'pattern': s.pattern or None, 'response_bytes': line.split()[0], 'response_ip': line.split()[3], 'icmp_seq': line.split()[5], 'ttl': line.split()[7], 'time_ms': line.split()[9]}
            return output_line
    elif ' bytes from ' in line:
        line = line.replace(',', ' ').replace('=', ' ')
        output_line = {'type': 'reply', 'destination_ip': s.destination_ip or None, 'sent_bytes': s.sent_bytes or None, 'pattern': s.pattern or None, 'bytes': line.split()[0], 'response_ip': line.split()[3], 'icmp_seq': line.split()[5], 'ttl': line.split()[7], 'time_ms': line.split()[9]}
        return output_line

def _linux_parse(line, s):
    if False:
        i = 10
        return i + 15
    '\n    Linux ping line parsing function.\n\n    Parameters:\n\n        line:        (string)        line of text data to parse\n        s:           (state object)  global state\n\n    Returns:\n\n        Dictionary. Raw structured data.\n    '
    output_line = {}
    if line.startswith('PING '):
        s.ipv4 = 'bytes of data' in line
        if s.ipv4 and line[5] not in string.digits:
            s.hostname = True
            line = line[:5] + 'nohost' + line[5:]
        elif s.ipv4 and line[5] in string.digits:
            s.hostname = False
        elif not s.ipv4 and ' (' in line:
            s.hostname = True
        else:
            s.hostname = False
        if s.ipv4 and (not s.hostname):
            (dst_ip, dta_byts) = (2, 3)
        elif s.ipv4 and s.hostname:
            (dst_ip, dta_byts) = (2, 3)
        elif not s.ipv4 and (not s.hostname):
            (dst_ip, dta_byts) = (2, 3)
        else:
            (dst_ip, dta_byts) = (3, 4)
        line = line.replace('(', ' ').replace(')', ' ')
        s.destination_ip = line.split()[dst_ip].lstrip('(').rstrip(')')
        s.sent_bytes = line.split()[dta_byts]
        return None
    if line.startswith('---'):
        s.footer = True
        return None
    if s.footer:
        m = re.search('(\\d+) packets transmitted', line)
        if m:
            s.packets_transmitted = m.group(1)
        m = re.search('(\\d+) received,', line)
        if m:
            s.packets_received = m.group(1)
        m = re.search('[+](\\d+) duplicates', line)
        if m:
            s.duplicates = m.group(1)
        m = re.search('[+](\\d+) corrupted', line)
        if m:
            s.corrupted = m.group(1)
        m = re.search('[+](\\d+) errors', line)
        if m:
            s.errors = m.group(1)
        m = re.search('([\\d\\.]+)% packet loss', line)
        if m:
            s.packet_loss_percent = m.group(1)
        m = re.search('time (\\d+)ms', line)
        if m:
            s.time_ms = m.group(1)
        m = re.search('rtt min\\/avg\\/max\\/mdev += +([\\d\\.]+)\\/([\\d\\.]+)\\/([\\d\\.]+)\\/([\\d\\.]+) ms', line)
        if m:
            s.round_trip_ms_min = m.group(1)
            s.round_trip_ms_avg = m.group(2)
            s.round_trip_ms_max = m.group(3)
            s.round_trip_ms_stddev = m.group(4)
        output_line = {'type': 'summary', 'destination_ip': s.destination_ip or None, 'sent_bytes': s.sent_bytes or None, 'pattern': s.pattern or None, 'packets_transmitted': s.packets_transmitted or None, 'packets_received': s.packets_received or None, 'packet_loss_percent': s.packet_loss_percent, 'duplicates': s.duplicates or '0', 'errors': s.errors, 'corrupted': s.corrupted, 'time_ms': s.time_ms, 'round_trip_ms_min': s.round_trip_ms_min, 'round_trip_ms_avg': s.round_trip_ms_avg, 'round_trip_ms_max': s.round_trip_ms_max, 'round_trip_ms_stddev': s.round_trip_ms_stddev}
        return output_line
    if 'no answer yet for icmp_seq=' in line:
        timestamp = False
        isequence = 5
        if line[0] == '[':
            timestamp = True
            isequence = 6
        output_line = {'type': 'timeout', 'destination_ip': s.destination_ip or None, 'sent_bytes': s.sent_bytes or None, 'pattern': s.pattern or None, 'timestamp': line.split()[0].lstrip('[').rstrip(']') if timestamp else None, 'icmp_seq': line.replace('=', ' ').split()[isequence]}
        return output_line
    if ' bytes from ' in line:
        line = line.replace('(', ' ').replace(')', ' ').replace('=', ' ')
        if s.ipv4 and (not s.hostname):
            (bts, rip, iseq, t2l, tms) = (0, 3, 5, 7, 9)
        elif s.ipv4 and s.hostname:
            (bts, rip, iseq, t2l, tms) = (0, 4, 7, 9, 11)
        elif not s.ipv4 and (not s.hostname):
            (bts, rip, iseq, t2l, tms) = (0, 3, 5, 7, 9)
        elif not s.ipv4 and s.hostname:
            (bts, rip, iseq, t2l, tms) = (0, 4, 7, 9, 11)
        timestamp = False
        if line[0] == '[':
            timestamp = True
            (bts, rip, iseq, t2l, tms) = (bts + 1, rip + 1, iseq + 1, t2l + 1, tms + 1)
        output_line = {'type': 'reply', 'destination_ip': s.destination_ip or None, 'sent_bytes': s.sent_bytes or None, 'pattern': s.pattern or None, 'timestamp': line.split()[0].lstrip('[').rstrip(']') if timestamp else None, 'response_bytes': line.split()[bts], 'response_ip': line.split()[rip].rstrip(':'), 'icmp_seq': line.split()[iseq], 'ttl': line.split()[t2l], 'time_ms': line.split()[tms], 'duplicate': 'DUP!' in line}
        return output_line

@add_jc_meta
def parse(data, raw=False, quiet=False, ignore_exceptions=False):
    if False:
        i = 10
        return i + 15
    '\n    Main text parsing generator function. Returns an iterable object.\n\n    Parameters:\n\n        data:              (iterable)  line-based text data to parse\n                                       (e.g. sys.stdin or str.splitlines())\n\n        raw:               (boolean)   unprocessed output if True\n        quiet:             (boolean)   suppress warning messages if True\n        ignore_exceptions: (boolean)   ignore parsing exceptions if True\n\n    Returns:\n\n        Iterable of Dictionaries\n    '
    jc.utils.compatibility(__name__, info.compatible, quiet)
    streaming_input_type_check(data)
    s = _state()
    summary_obj = {}
    for line in data:
        try:
            streaming_line_input_type_check(line)
            output_line = {}
            if not line.strip():
                continue
            if line.startswith('WARNING: '):
                continue
            if line.startswith('PATTERN: '):
                s.pattern = line.strip().split(': ')[1]
                continue
            if not s.os_detected and line.strip().endswith('bytes of data.'):
                s.os_detected = True
                s.linux = True
            elif not s.os_detected and '-->' in line:
                s.os_detected = True
                s.bsd = True
            elif not s.os_detected and _ipv6_in(line) and line.strip().endswith('data bytes'):
                s.os_detected = True
                s.linux = True
            elif not s.os_detected and (not _ipv6_in(line)) and line.strip().endswith('data bytes'):
                s.os_detected = True
                s.bsd = True
            if s.os_detected and s.linux:
                output_line = _linux_parse(line, s)
                if output_line:
                    if output_line.get('type', None) == 'summary':
                        summary_obj = output_line
                        continue
            elif s.os_detected and s.bsd:
                output_line = _bsd_parse(line, s)
            else:
                raise ParseError('Could not detect ping OS')
            if output_line:
                yield (output_line if raw else _process(output_line))
            else:
                continue
        except Exception as e:
            yield raise_or_yield(ignore_exceptions, e, line)
    try:
        if summary_obj:
            yield (summary_obj if raw else _process(summary_obj))
    except Exception as e:
        yield raise_or_yield(ignore_exceptions, e, str(summary_obj))