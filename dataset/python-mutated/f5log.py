__author__ = 'James Deucker <me@bitwisecook.org>'
__version__ = '1.0.10'
from datetime import datetime, timedelta
from ipaddress import ip_address
import re
import traceback
from typing import Any, Dict, Optional
from visidata import Path, VisiData, Sheet, date, AttrColumn, vd, Column, CellColorizer, RowColorizer

class hexint(int):

    def __new__(cls, value, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return super(cls, cls).__new__(cls, value, base=16)

    def __str__(self):
        if False:
            return 10
        return hex(self)

class delta_t(int):

    def __new__(cls, value, *args, **kwargs):
        if False:
            while True:
                i = 10
        return super(cls, cls).__new__(cls, value, base=10)
vd.addType(hexint, icon='ⓧ', formatter=lambda fmt, num: str(num))
vd.addType(delta_t, icon='⇥', formatter=lambda fmt, delta: str(timedelta(seconds=delta)))
vd.theme_option('color_f5log_mon_up', 'green', 'color of f5log monitor status up')
vd.theme_option('color_f5log_mon_down', 'red', 'color of f5log monitor status down')
vd.theme_option('color_f5log_mon_unknown', 'blue', 'color of f5log monitor status unknown')
vd.theme_option('color_f5log_mon_checking', 'magenta', 'color of monitor status checking')
vd.theme_option('color_f5log_mon_disabled', 'black', 'color of monitor status disabled')
vd.theme_option('color_f5log_logid_alarm', 'red', 'color of alarms')
vd.theme_option('color_f5log_logid_warn', 'yellow', 'color of warnings')
vd.theme_option('color_f5log_logid_notice', 'cyan', 'color of notice')
vd.theme_option('color_f5log_logid_info', 'green', 'color of info')
vd.option('f5log_object_regex', None, 'A regex to perform on the object name, useful where object names have a structure to extract. Use the (?P<foo>...) named groups form to get column names.', help=vd.help_regex)
vd.option('f5log_log_year', None, 'Override the default year used for log parsing. Use all four digits of the year (e.g., 2022). By default (None) use the year from the ctime of the file, or failing that the current year.')
vd.option('f5log_log_timezone', 'UTC', 'The timezone the source file is in, by default UTC.')

class F5LogSheet(Sheet):

    class F5LogRow:

        def __init__(self, msg: str=None, timestamp: datetime=None, host: str=None, level: str=None, process: str=None, proc_pid: int=None, logid1: hexint=None, logid2: hexint=None, message: str=None, kv: Optional[Dict[str, Any]]=None, **kwargs):
            if False:
                return 10
            self._data = {'msg': msg, 'timestamp': timestamp, 'host': host, 'level': level, 'process': process, 'proc_pid': proc_pid, 'logid1': logid1, 'logid2': logid2, 'message': message, 'kv': kv, **kwargs}

        def __getattr__(self, item):
            if False:
                for i in range(10):
                    print('nop')
            return self._data.get(item, self._data['kv'].get(item) if self._data['kv'] else None)
    _months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    _proto = {6: 'tcp', 17: 'udp', 132: 'sctp'}
    rowtype = 'logs'
    columns = [AttrColumn('rawmsg', type=str, width=0), AttrColumn('timestamp', type=date), AttrColumn('host', type=str), AttrColumn('level', type=str), AttrColumn('process', type=str, width=10), AttrColumn('proc_pid', type=int, width=7), AttrColumn('logid1', type=hexint), AttrColumn('logid2', type=hexint), AttrColumn('message', type=str, width=90), AttrColumn('object', type=str, width=50)]
    re_f5log = re.compile('^(?:(?:audit|gtm|ltm|security|tmm|user|\\<\\d+\\>)\\s+)?(?:(?P<date1>[A-Z][a-z]{2}\\s+\\d{1,2}\\s+\\d{2}:\\d{2}:\\d{2})|(?P<date2>\\d+-\\d+-\\d+T\\d+:\\d+:\\d+[+-]\\d+:\\d+)|(?P<date3>\\d+-\\d+\\s\\d{2}:\\d{2}:\\d{2}))\\s+(?P<host>\\S+)\\s+(?:(?P<level>\\S+)\\s+(?:(?P<process>[a-z0-9_()-]+\\s?)\\[(?P<pid>\\d+)\\]:\\s+)?(?:(?P<logid1>[0-9a-f]{8}):(?P<logid2>[0-9a-f]):\\s+)?)?(?P<message>.*)$')
    re_ltm_irule = re.compile('(?:(?P<irule_msg>TCL\\serror|Rule|Pending\\srule):?\\s(?P<irule>\\S+)\\s\\<(?P<event>[A-Z_0-9]+)\\>(?:\\s-\\s|:\\s|\\s)?)(?P<message>aborted\\sfor\\s(?P<srchost>\\S+)\\s->\\s(?P<dsthost>\\S+)|.*)')
    re_ltm_pool_mon_status_msg = re.compile('^(Pool|Node)\\s(?P<poolobj>\\S+)\\s(member|address)\\s(?P<poolmemberobj>\\S+)\\smonitor\\sstatus\\s(?P<newstatus>.+)\\.\\s\\[\\s(?:(?:(?:(?P<monitorobj>\\S+):\\s(?P<monitorstatus>\\w+)(?:;\\slast\\serror:\\s\\S*\\s?(?P<lasterr>[^\\]]*))?)?(?:,\\s)?)+)?\\s]\\s*(?:\\[\\swas\\s(?P<prevstatus>.+)\\sfor\\s(?P<durationhr>-?\\d+)hrs?:(?P<durationmin>-?\\d+)mins?:(?P<durationsec>-?\\d+)sec\\s\\])?$')
    re_ltm_ip_msg = re.compile('(?:.*?)(?P<ip1>\\d+\\.\\d+\\.\\d+\\.\\d+)(?:[:.](?P<port1>\\d+))?(?:(?:\\s->\\s|:)(?P<ip2>\\d+\\.\\d+\\.\\d+\\.\\d+)(?:[:.](?P<port2>\\d+))?)?(?:\\smonitor\\sstatus\\s(?P<mon_status>\\w+)\\.\\s\\[[^]]+\\]\\s+\\[\\swas\\s(?P<prev_status>\\w+)\\sfor\\s((?P<durationhr>-?\\d+)hrs?:(?P<durationmin>-?\\d+)mins?:(?P<durationsec>-?\\d+)secs?)\\s\\]|\\.?(?:.*))')
    re_ltm_conn_error = re.compile('^(?:\\(null\\sconnflow\\):\\s)?Connection\\serror:\\s(?P<func>[^:]+):(?P<funcloc>[^:]+):\\s(?P<error>.*)\\s?\\((?P<errno>\\d+)\\)(?:\\s(?P<errormsg>.*))$')
    re_ltm_cert_expiry = re.compile("Certificate\\s'(?P<cert_cn>.*)'\\sin\\sfile\\s(?P<file>\\S+)\\s(?P<message>will\\sexpire|expired)\\son\\s(?P<date1>[A-Z][a-z]{2}\\s+\\d{1,2}\\s+\\d{2}:\\d{2}:\\d{2}\\s\\d+\\s\\S+)")
    re_gtm_monitor = re.compile('^(?:SNMP_TRAP:\\s)?(?P<objtype>VS|Pool|Monitor|Wide\\sIP|Server|Data\\scenter|Prober\\sPool|Box)\\s\\(?(?P<object>\\S+?)\\)?\\s(?:member\\s\\(?(?P<pool_member>\\S+?)\\)?\\s)?(?:\\(ip(?::port)?=(?P<ipport>[^\\)]+)\\)\\s)?(?:\\(Server\\s(?P<server>[^\\)]+)\\)\\s)?(?:state\\schange\\s)?(?:(?P<prev_status>\\w+)\\s-->\\s)?(?P<new_status>\\w+)(?:(?:\\s\\(\\s?)(?P<msg>(?:(?P<type>\\w+)\\s(?P<monitor_object>\\S+)\\s:\\s)?state:\\s(?P<state>\\S+)|.*)\\))?$')
    re_gtm_monitor_instance = re.compile('^Monitor\\sinstance\\s(?P<object>\\S+)\\s(?P<monip>\\S+)\\s(?P<prevstatus>\\S+)\\s-->\\s(?P<newstatus>\\S+)\\sfrom\\s(?P<srcgtm>\\S+)\\s\\((?:state:?\\s)?(?P<state>.*)\\)$')
    re_ltm_poolnode_abled = re.compile('^(?P<objtype>Pool|Node|Monitor)\\s(?P<object>\\S+)\\s(?:address|member|instance)\\s(?P<member>\\S+)\\s(session\\sstatus|has\\sbeen)\\s(?P<status>.+)\\.$')
    re_ltm_no_shared_ciphers = re.compile('^(?P<msg>No\\sshared\\sciphers\\sbetween\\sSSL\\speers)\\s(?P<srchost>\\d+\\.\\d+\\.\\d+\\.\\d+|[0-9a-f:]+)\\.(?P<srcport>\\d+)\\:(?P<dsthost>\\d+\\.\\d+\\.\\d+\\.\\d+|[0-9a-f:]+)\\.(?P<dstport>\\d+)\\.$')
    re_ltm_http_process_state = re.compile('^http_process_state_(?P<httpstate>\\S+)\\s-\\sInvalid\\saction:0x(?P<actionid>[a-f0-9]+)\\s(?P<msg>.*?)\\s*(?P<sidea>(?:server|client)side)\\s\\((?P<src>\\S+)\\s->\\s(?P<vsdst>\\S+)\\)\\s+(?:(?P<sideb>(?:server|client)side)\\s)?(?:\\((?P<poolsrc>\\S+)\\s->\\s(?P<dst>\\S+)\\)|\\(\\(null\\sconnflow\\)\\))\\s\\((?P<sideaa>\\S+)\\sside:\\svip=(?P<vs>\\S+)\\sprofile=(?P<profile>\\S+)\\s(?:pool=(?P<pool>\\S+)\\s)?(?P<sideaaa>\\S+)_ip=(?P<sideasrc>\\S+)\\)$')
    re_ltm_http_header_exceeded = re.compile('^(?P<msg>HTTP\\sheader\\s(?:count|\\((?P<size>\\d+)\\))\\sexceeded\\smaximum\\sallowed\\s(?P<type>count|size)\\sof\\s(?P<limit>\\d+))\\s\\((?P<side>\\S+)\\sside:\\svip=(?P<object>\\S+)\\sprofile=(?P<profile>\\S+)\\spool=(?P<pool>\\S+)\\s(?P<sideip>client|server)_ip=(?P<sidehost>.*)\\)$')
    f5log_mon_colors = {('monitor_status', 'down'): 'color_f5log_mon_down', ('monitor_status', 'up'): 'color_f5log_mon_up', ('monitor_status', 'enabled'): 'color_f5log_mon_up', ('monitor_status', 'forced disabled'): 'color_f5log_mon_disabled', ('monitor_status', 'node disabled'): 'color_f5log_mon_disabled', ('monitor_status', 'checking'): 'color_f5log_mon_checking', ('new_status', 'available'): 'color_f5log_mon_up', ('new_status', 'unavailable'): 'color_f5log_mon_down', ('new_status', 'up'): 'color_f5log_mon_up', ('new_status', 'down'): 'color_f5log_mon_down', ('new_status', 'green'): 'color_f5log_mon_up', ('new_status', 'red'): 'color_f5log_mon_down', ('new_status', 'now has available members'): 'color_f5log_mon_up', ('new_status', 'no members available'): 'color_f5log_mon_down', ('new_status', 'blue'): 'color_f5log_mon_unknown', ('new_status', 'checking'): 'color_f5log_mon_checking', ('new_status', 'unchecked'): 'color_f5log_mon_unknown', ('new_status', 'node down'): 'color_f5log_mon_disabled', ('new_status', 'forced down'): 'color_f5log_mon_disabled', ('new_status', 'disabled'): 'color_f5log_mon_disabled', ('prev_status', 'available'): 'color_f5log_mon_up', ('prev_status', 'unavailable'): 'color_f5log_mon_down', ('prev_status', 'up'): 'color_f5log_mon_up', ('prev_status', 'down'): 'color_f5log_mon_down', ('prev_status', 'green'): 'color_f5log_mon_up', ('prev_status', 'red'): 'color_f5log_mon_down', ('prev_status', 'now has available members'): 'color_f5log_mon_up', ('prev_status', 'no members available'): 'color_f5log_mon_down', ('prev_status', 'blue'): 'color_f5log_mon_unknown', ('prev_status', 'checking'): 'color_f5log_mon_checking', ('prev_status', 'unchecked'): 'color_f5log_mon_unknown', ('prev_status', 'node down'): 'color_f5log_mon_disabled', ('prev_status', 'forced down'): 'color_f5log_mon_disabled', ('prev_status', 'disabled'): 'color_f5log_mon_disabled'}

    def colorizeMonitors(sheet, col: Column, row: F5LogRow, value):
        if False:
            return 10
        if row is None or col is None:
            return None
        return sheet.f5log_mon_colors.get((col.name, value.value), None)
    f5log_warn_logid = {'01010013': 'color_f5log_logid_notice', '01010029': 'color_f5log_logid_warn', '01010038': 'color_f5log_logid_warn', '01010201': 'color_f5log_logid_warn', '01010281': 'color_f5log_logid_warn', '01070333': 'color_f5log_logid_warn', '01070596': 'color_f5log_logid_alarm', '010c0018': 'color_f5log_logid_warn', '010c0019': 'color_f5log_logid_info', '010c003e': 'color_f5log_logid_alarm', '010c003f': 'color_f5log_logid_alarm', '010c0044': 'color_f5log_logid_warn', '010c0052': 'color_f5log_logid_warn', '010c0053': 'color_f5log_logid_info', '010c0054': 'color_f5log_logid_alarm', '010c0055': 'color_f5log_logid_alarm', '010c0057': 'color_f5log_logid_info', '010e0001': 'color_f5log_logid_alarm', '010e0004': 'color_f5log_logid_alarm', '01340011': 'color_f5log_logid_warn', '01390002': 'color_f5log_logid_notice', '01140029': 'color_f5log_logid_alarm', '01140030': 'color_f5log_logid_warn', '01140045': 'color_f5log_logid_alarm', '01190004': 'color_f5log_logid_alarm', '011ae0f3': 'color_f5log_logid_alarm', '011e0002': 'color_f5log_logid_alarm', '011e0003': 'color_f5log_logid_alarm', '011f0005': 'color_f5log_logid_warn', '014f0004': 'color_f5log_logid_warn'}

    def colorizeRows(sheet, col: Column, row: F5LogRow, value):
        if False:
            while True:
                i = 10
        if row is None or col is None:
            return None
        if row.logid1 is None and row.message is not None and row.message.startswith('boot_marker'):
            return 'color_f5log_logid_notice'
        return sheet.f5log_warn_logid.get(row.logid1, None)

    @staticmethod
    def split_audit_bigip_tmsh_audit(msg):
        if False:
            i = 10
            return i + 15
        e = msg[8:].split('=', maxsplit=6)
        for (ee, ne) in zip(e, e[1:]):
            yield {ee[ee.rfind(' ') + 1:]: ne[:ne.rfind(' ')]}

    @staticmethod
    def split_audit_scriptd_run_script(msg):
        if False:
            for i in range(10):
                print('nop')
        e = msg[8:].split('=')
        for (ee, ne) in zip(e, e[1:]):
            yield {ee[ee.rfind(' ') + 1:]: ne[:ne.rfind(' ')].strip('"')}

    @staticmethod
    def split_audit_mcpd_mcp_error(msg):
        if False:
            i = 10
            return i + 15
        msg = msg[8:]
        status = None
        status_loc = msg.rfind('[Status=')
        if status_loc >= 0:
            status = msg[status_loc + 1:-1]
            msg = msg[:status_loc - 1]
            yield {'status': status.split('=', maxsplit=1)[1]}
        cmd_data_loc = msg.find(' { ')
        if cmd_data_loc >= 0:
            cmd_data = msg[cmd_data_loc + 1:]
            (msg, cmd) = msg[:cmd_data_loc].rsplit(' ', maxsplit=1)
            msg = msg[:-2]
            object = cmd_data.split('"', maxsplit=2)
            if len(object) == 3:
                yield {'object': object[1]}
            yield {'command': cmd, 'cmd_data': cmd_data}
        e = msg.split(' - ')
        for ee in e[0].split(','):
            ee = ee.strip().split(' ')
            if ee[0].startswith('tmsh-pid-'):
                yield {ee[0][:ee[0].rfind('-')]: int(ee[0][ee[0].rfind('-') + 1:])}
            elif len(ee) == 1:
                yield {ee[0]: None}
            else:
                yield {ee[0]: ee[1]}
        for ee in e[1:]:
            ee = ee.strip().split(' ', maxsplit=1)
            if ee[0] == 'transaction':
                yield {'transaction': int(ee[1][1:].split('-')[0])}
                yield {'transaction_step': int(ee[1][1:].split('-')[1])}
            elif ee[0] == 'object':
                yield {'object_id': ee[1]}
            else:
                try:
                    yield {ee[0]: ee[1]}
                except IndexError:
                    yield {ee[0]: None}

    @staticmethod
    def split_ltm_pool_mon_status(msg):
        if False:
            for i in range(10):
                print('nop')
        m = F5LogSheet.re_ltm_pool_mon_status_msg.match(msg)
        if m is None:
            return
        m = m.groupdict()
        if m.get('durationhr') and m.get('durationmin') and m.get('durationsec'):
            duration = timedelta(hours=int(m.get('durationhr')), minutes=int(m.get('durationmin')), seconds=int(m.get('durationsec'))).total_seconds()
        else:
            duration = None
        dst = m.get('poolmemberobj')
        if dst:
            dst = dst.split('/')[-1]
            if '.' in dst and len(dst.split(':')) == 2:
                (dsthost, dstport) = dst.split(':')
            elif '.' in dst and len(dst.split(':')) == 1:
                (dsthost, dstport) = (dst, None)
            else:
                (dsthost, dstport) = dst.rsplit(':', maxsplit=1)
            try:
                dsthost = ip_address(dsthost)
            except ValueError:
                dsthost = None
            try:
                dstport = int(dstport)
            except (ValueError, TypeError):
                dstport = None
        else:
            (dsthost, dstport) = (None, None)
        yield {'object': m.get('poolobj'), 'objtype': 'pool', 'pool_member': m.get('poolmemberobj'), 'monitor': m.get('monitorobj'), 'dsthost': dsthost, 'dstport': dstport, 'monitor_status': m.get('monitorstatus'), 'new_status': m.get('newstatus'), 'prev_status': m.get('prevstatus'), 'last_error': m.get('lasterr'), 'duration_s': duration}

    @staticmethod
    def split_ltm_poolnode_mon_abled(msg):
        if False:
            i = 10
            return i + 15
        m = F5LogSheet.re_ltm_poolnode_abled.match(msg)
        if m is None:
            return
        m = m.groupdict()
        yield {'object': m.get('object'), 'objtype': m.get('objtype').lower(), 'pool_member': m.get('member'), 'monitor_status': m.get('status')}

    @staticmethod
    def split_ltm_pool_has_no_avail_mem(msg):
        if False:
            return 10
        yield {'object': msg.split(' ')[-1], 'objtype': 'pool', 'new_status': 'no members available', 'prev_status': None}

    @staticmethod
    def split_ltm_pool_has_avail_mem(msg):
        if False:
            while True:
                i = 10
        yield {'object': msg.split(' ')[1], 'objtype': 'pool', 'new_status': 'now has available members', 'prev_status': None}

    @staticmethod
    def split_ltm_rule(msg):
        if False:
            return 10
        m = F5LogSheet.re_ltm_irule.match(msg)
        if m is None:
            return
        m = m.groupdict()
        yield {'object': m.get('irule'), 'objtype': 'rule', 'irule_event': m.get('event'), 'irule_msg': m.get('irule_msg'), 'msg': m.get('message')}
        if m.get('message', '').startswith('aborted for'):
            src = m.get('srchost')
            if src and len(src.split(':')) == 2:
                (srchost, srcport) = src.split(':')
            else:
                (srchost, srcport) = src.split('.')
            dst = m.get('dsthost')
            if dst and len(dst.split(':')) == 2:
                (dsthost, dstport) = dst.split(':')
            else:
                (dsthost, dstport) = dst.split('.')
            yield {'srchost': ip_address(srchost), 'srcport': int(srcport), 'dsthost': ip_address(dsthost), 'dstport': int(dstport)}

    @staticmethod
    def split_ltm_rule_missing_datagroup(msg):
        if False:
            while True:
                i = 10
        if 'error: Unable to find value_list' in msg:
            m = msg.split(' ', maxsplit=12)
            yield {'object': m[1].strip('[').strip(']'), 'objtype': 'rule', 'msg': 'error: Unable to find value_list', 'missing_dg': m[7].strip('('), 'funcloc': int(m[11].strip(':')), 'error': m[12]}
        else:
            m = msg.split(' ', maxsplit=5)
            yield {'object': m[1].strip('[').strip(']'), 'objtype': 'rule', 'msg': m[5].split(']', maxsplit=1)[0].strip('[]'), 'error': m[5].split(']', maxsplit=1)[1], 'funcloc': int(m[3].split(':')[1])}

    @staticmethod
    def split_ltm_cert_expiry(msg):
        if False:
            return 10
        m = F5LogSheet.re_ltm_cert_expiry.match(msg)
        if m is None:
            return
        m = m.groupdict()
        yield {'cert_cn': m.get('cert_cn'), 'object': m.get('file'), 'objtype': 'ssl-cert', 'date': datetime.strptime(m.get('date1').replace('  ', ' '), '%b %d %H:%M:%S %Y %Z') if m.get('date1') is not None else None, 'msg': m.get('message')}

    @staticmethod
    def split_ltm_connection_error(msg):
        if False:
            return 10
        m = F5LogSheet.re_ltm_conn_error.match(msg)
        if m is None:
            return
        m = m.groupdict()
        yield {'func': m.get('func'), 'funcloc': m.get('funcloc'), 'error': m.get('error'), 'errno': m.get('errno'), 'errmsg': m.get('errormsg')}

    @staticmethod
    def split_ltm_virtual_status(msg):
        if False:
            i = 10
            return i + 15
        m = msg.split(' ')
        if m[0] == 'SNMP_TRAP:':
            yield {'object': m[2], 'objtype': 'vs', 'new_status': m[-1], 'prev_status': None}
        else:
            yield {'object': m[1], 'objtype': 'vs', 'new_status': m[-1], 'prev_status': None}

    @staticmethod
    def split_ltm_virtual_address_status_or_irule_profile_err(msg):
        if False:
            while True:
                i = 10
        m = msg.split(' ')
        if 'event in rule' in msg:
            yield {'object': m[4].strip('()'), 'objtype': 'rule', 'irule_event': m[0], 'msg': f"event in rule {' '.join(m[5:-2])}", 'target_obj': m[-1].strip('().'), 'target_objtype': m[-2].replace('virtual-server', 'vs')}
        else:
            yield {'object': m[2], 'objtype': 'virtual address', 'new_status': m[9].lower().strip('.'), 'prev_status': m[7].lower()}

    @staticmethod
    def split_ltm_ssl_handshake_fail(msg):
        if False:
            i = 10
            return i + 15
        src = msg.split(' ')[5]
        if len(src.split(':')) == 2:
            (srchost, srcport) = src.split(':')
        else:
            (srchost, srcport) = src.split('.')
        dst = msg.split(' ')[7]
        if len(dst.split(':')) == 2:
            (dsthost, dstport) = dst.split(':')
        else:
            (dsthost, dstport) = dst.split('.')
        yield {'srchost': ip_address(srchost), 'srcport': int(srcport), 'dsthost': ip_address(dsthost), 'dstport': int(dstport)}

    @staticmethod
    def split_ltm_shared_ciphers(msg):
        if False:
            while True:
                i = 10
        m = F5LogSheet.re_ltm_no_shared_ciphers.match(msg)
        if m is None:
            return
        m = m.groupdict()
        yield {'srchost': ip_address(m.get('srchost')), 'srcport': int(m.get('srcport')), 'dsthost': ip_address(m.get('dsthost')), 'dstport': int(m.get('dstport'))}

    @staticmethod
    def split_ltm_rst_reason(msg):
        if False:
            return 10
        m = msg.split(' ', maxsplit=7)
        (src, dst) = (m[3].strip(','), m[5].strip(','))
        (reasonc1, reasonc2) = m[6].split(':')
        if len(src.split(':')) == 2:
            (srchost, srcport) = src.split(':')
        else:
            (srchost, srcport) = src.rsplit(':', maxsplit=1)
        if len(dst.split(':')) == 2:
            (dsthost, dstport) = dst.split(':')
        else:
            (dsthost, dstport) = dst.rsplit(':', maxsplit=1)
        yield {'srchost': ip_address(srchost), 'srcport': int(srcport) if srcport else None, 'dsthost': ip_address(dsthost), 'dstport': int(dstport) if dstport else None, 'rst_reason_code1': hexint(reasonc1[3:]), 'rst_reason_code2': hexint(reasonc2[:-1]), 'rst_reason': m[7]}

    @staticmethod
    def split_ltm_inet_port_exhaust(msg):
        if False:
            i = 10
            return i + 15
        m = msg.split(' ')
        (srchost, dst) = (m[-5], m[-3])
        if len(dst.split(':')) == 2:
            (dsthost, dstport) = dst.split(':')
        else:
            (dsthost, dstport) = dst.rsplit(':', maxsplit=1)
        yield {'msg': ' '.join(m[:5] if len(m) == 11 else m[:3]), 'srchost': ip_address(srchost), 'dsthost': ip_address(dsthost), 'dstport': int(dstport), 'proto': F5LogSheet._proto[int(m[-1].strip(')'))]}

    @staticmethod
    def split_ltm_conn_limit_reached(msg):
        if False:
            for i in range(10):
                print('nop')
        m = msg.split(' ', maxsplit=11)
        (src, dst) = (m[4], m[6].strip(','))
        if len(src.split(':')) == 2:
            (srchost, srcport) = src.split(':')
        else:
            (srchost, srcport) = src.rsplit('.', maxsplit=1)
        if len(dst.split(':')) == 2:
            (dsthost, dstport) = dst.split(':')
        else:
            (dsthost, dstport) = dst.rsplit('.', maxsplit=1)
        yield {'msg': m[-1], 'object': m[10].strip(':'), 'objtype': m[9].lower(), 'srchost': ip_address(srchost), 'srcport': int(dstport), 'dsthost': ip_address(dsthost), 'dstport': int(dstport), 'proto': m[8].strip(',').lower()}

    @staticmethod
    def split_ltm_syncookie_threshold(msg):
        if False:
            i = 10
            return i + 15
        m = msg.split(' ')
        dst = m[-1]
        if len(dst.split(':')) == 2:
            (dsthost, dstport) = dst.split(':')
        else:
            (dsthost, dstport) = dst.rsplit('.', maxsplit=1)
        yield {'msg': 'Syncookie threshold exceeded', 'threshold': int(m[2]), 'objtype': m[-3].lower(), 'dsthost': ip_address(dsthost), 'dstport': int(dstport)}

    @staticmethod
    def split_ltm_sweeper_active2(msg):
        if False:
            for i in range(10):
                print('nop')
        m = msg.split(' ')
        yield {'policy': m[3], 'mode': m[4], 'object': m[7].strip(').'), 'objtype': m[6].strip('('), 'msg': ' '.join(m[8:]).strip('()')}

    @staticmethod
    def split_ltm_sweeper_active3(msg):
        if False:
            print('Hello World!')
        m = msg.split(' ')
        yield {'policy': m[3], 'object': m[6].strip(').'), 'objtype': m[5].strip('('), 'msg': ' '.join(m[7:]).strip('()')}

    @staticmethod
    def split_ltm_dns_failed_xfr_rcode(msg):
        if False:
            while True:
                i = 10
        m = msg.split(' ')
        yield {'msg': ' '.join([m[0], *m[4:]]), 'zone': m[3]}

    @staticmethod
    def split_ltm_dns_failed_rr(msg):
        if False:
            return 10
        m = msg.rsplit(' ', maxsplit=3)
        yield {'msg': m[0], 'zone': m[-1].strip('.')}

    @staticmethod
    def split_ltm_dns_failed_xfr(msg):
        if False:
            print('Hello World!')
        m = msg.split(' ')
        src = m[6].strip(',')
        yield {'msg': ' '.join(m[:3]) + ', ' + ' '.join(m[-2:]), 'srchost': ip_address(src), 'zone': m[4]}

    @staticmethod
    def split_ltm_dns_handling_notify(msg):
        if False:
            i = 10
            return i + 15
        m = msg.split(' ')
        yield {'msg': ' '.join(m[:2]), 'zone': m[4].rstrip('.')}

    @staticmethod
    def split_ltm_dns_axfr_succeeded_1f(msg):
        if False:
            i = 10
            return i + 15
        m = msg.split(' ')
        yield {'msg': ' '.join([*m[:3], m[-1]]), 'srchost': ip_address(m[6]), 'zone': m[4]}

    @staticmethod
    def split_ltm_dns_axfr_succeeded_2c(msg):
        if False:
            while True:
                i = 10
        m = msg.split(' ')
        yield {'msg': ' '.join([*m[:1], m[-1]]), 'srchost': m[10], 'zone': m[4], 'serial': m[8]}

    @staticmethod
    def split_ltm_dns_ignoring_tfer(msg):
        if False:
            while True:
                i = 10
        m = msg.split(' ')
        yield {'msg': ' '.join([*m[:2], *m[-3:]]), 'srchost': m[6].strip(';'), 'zone': m[4]}

    @staticmethod
    def split_ltm_http_process_state(msg):
        if False:
            print('Hello World!')
        m = F5LogSheet.re_ltm_http_process_state.match(msg)
        if not m:
            return
        m = m.groupdict()
        (src, vsdst, backendsrc, dst) = (m.get('src'), m.get('vsdst'), m.get('poolsrc'), m.get('dst'))
        if src:
            if len(src.split(':')) == 2:
                (srchost, srcport) = src.split(':')
            else:
                (srchost, srcport) = src.rsplit('.', maxsplit=1)
        else:
            (srchost, srcport) = (None, None)
        if dst:
            if len(dst.split(':')) == 2:
                (dsthost, dstport) = dst.split(':')
            else:
                (dsthost, dstport) = dst.rsplit('.', maxsplit=1)
        else:
            (dsthost, dstport) = (None, None)
        if vsdst:
            if len(vsdst.split(':')) == 2:
                (vsdsthost, vsdstport) = vsdst.split(':')
            else:
                (vsdsthost, vsdstport) = vsdst.rsplit('.', maxsplit=1)
        else:
            (vsdsthost, vsdstport) = (None, None)
        if backendsrc:
            if len(backendsrc.split(':')) == 2:
                (backendsrchost, backendsrcport) = backendsrc.split(':')
            else:
                (backendsrchost, backendsrcport) = backendsrc.rsplit('.', maxsplit=1)
        else:
            (backendsrchost, backendsrcport) = (None, None)
        yield {'object': m.get('vs'), 'msg': m.get('msg'), 'srchost': ip_address(srchost) if srchost else None, 'srcport': int(srcport) if srcport else None, 'dsthost': ip_address(dsthost) if dsthost else None, 'dstport': int(dstport) if dstport else None, 'pool': m.get('pool'), 'profile': m.get('profile'), 'httpstate': m.get('httpstate'), 'actionid': hexint(m.get('actionid')), 'sidea': m.get('sidea'), 'vsdsthost': ip_address(vsdsthost) if vsdsthost else None, 'vsdstport': int(vsdstport) if vsdstport else None, 'backendsrchost': ip_address(backendsrchost) if backendsrchost else None, 'backendsrcport': int(backendsrcport) if backendsrcport else None, 'sideb': m.get('sideb'), 'sideasrc': ip_address(m.get('sideasrc')) if m.get('sideasrc') else None}

    @staticmethod
    def split_ltm_http_header_exceeded(msg):
        if False:
            for i in range(10):
                print('nop')
        m = F5LogSheet.re_ltm_http_header_exceeded.match(msg)
        if not m:
            return
        m = m.groupdict()
        host = m.get('sidehost')
        yield {'object': m.get('object'), 'msg': m.get('msg'), 'srchost': ip_address(host) if host and m.get('sideip') == 'client' else None, 'dsthost': ip_address(host) if host and m.get('sideip') == 'server' else None, 'pool': m.get('pool'), 'profile': m.get('profile'), 'size': int(m.get('size')) if m.get('size') else None, 'limit': int(m.get('limit')) if m.get('limit') else None}

    @staticmethod
    def split_gtm_monitor(msg):
        if False:
            while True:
                i = 10
        m = F5LogSheet.re_gtm_monitor.match(msg)
        if m is None:
            return
        m = m.groupdict()
        dst = m.get('ipport')
        if dst:
            if len(dst.split('.')) == 4:
                if ':' in dst:
                    (dsthost, dstport) = dst.split(':')
                else:
                    (dsthost, dstport) = (dst, None)
            elif '.' in dst:
                (dsthost, dstport) = dst.rsplit('.', maxsplit=1)
            else:
                (dsthost, dstport) = (dst, None)
        else:
            (dsthost, dstport) = (None, None)
        yield {'objtype': m.get('objtype').lower() if m.get('objtype') else None, 'object': m.get('object'), 'pool_member': m.get('pool_member'), 'monitor_object': m.get('monitor_object'), 'dsthost': ip_address(dsthost) if dsthost else None, 'dstport': int(dstport) if dstport else None, 'server': m.get('server'), 'new_status': m.get('new_status').lower() if m.get('new_status') else None, 'prev_status': m.get('prev_status').lower() if m.get('prev_status') else None, 'msg': m.get('msg'), 'type': m.get('type').lower() if m.get('type') in m else None, 'state': m.get('state')}

    @staticmethod
    def split_gtm_monitor_instance(msg):
        if False:
            print('Hello World!')
        m = F5LogSheet.re_gtm_monitor_instance.match(msg)
        if m is None:
            return
        m = m.groupdict()
        if m.get('monip'):
            if len(m.get('monip').split(':')) == 2:
                (dsthost, dstport) = m.get('monip').split(':')
            else:
                (dsthost, dstport) = m.get('monip').rsplit(':', maxsplit=1)
        else:
            (dsthost, dstport) = (None, None)
        yield {'object': m.get('object'), 'objtype': 'monitor', 'dsthost': ip_address(dsthost) if dsthost else None, 'dstport': int(dstport) if dstport else None, 'new_status': m.get('newstatus', '').lower(), 'prev_status': m.get('prevstatus', '').lower(), 'src_gtm': m.get('srcgtm'), 'state': m.get('state').lower()}

    @staticmethod
    def split_gtm_syncgroup_change(msg):
        if False:
            while True:
                i = 10
        m = msg.split(' ')
        yield {'object': m[3], 'srchost': ip_address(m[4].strip('()')), 'syncgroup': m[-1], 'msg': f'BIG-IP GTM {m[5]} sync group'}

    @staticmethod
    def split_gtm_changed_state(msg):
        if False:
            print('Hello World!')
        m = msg.split(' ')
        yield {'msg': f'{m[0]} changed state', 'new_status': m[6].lower().strip('.'), 'prev_status': m[4].lower()}

    @staticmethod
    def split_tmm_address_conflict(msg):
        if False:
            for i in range(10):
                print('nop')
        m = msg.split(' ')
        dsthost = m[4]
        yield {'object': ' '.join(m[7:]), 'objtype': 'address', 'dsthost': ip_address(dsthost), 'dstmac': m[5].strip('()')}
    splitters = {16842792: split_ltm_pool_has_no_avail_mem.__func__, 16842808: split_ltm_syncookie_threshold.__func__, 16843265: split_ltm_inet_port_exhaust.__func__, 16843393: split_ltm_inet_port_exhaust.__func__, 16843297: split_ltm_pool_has_avail_mem.__func__, 17236305: split_ltm_rule_missing_datagroup.__func__, 17237015: split_audit_mcpd_mcp_error.__func__, 17237561: split_ltm_poolnode_mon_abled.__func__, 17237569: split_ltm_poolnode_mon_abled.__func__, 17238023: split_ltm_poolnode_mon_abled.__func__, 17238024: split_ltm_poolnode_mon_abled.__func__, 17237799: split_ltm_pool_mon_status.__func__, 17237800: split_ltm_pool_mon_status.__func__, 17237560: split_ltm_pool_mon_status.__func__, 17237568: split_ltm_pool_mon_status.__func__, 17241729: split_ltm_virtual_status.__func__, 17241730: split_ltm_virtual_status.__func__, 17242386: split_ltm_virtual_address_status_or_irule_profile_err.__func__, 17242387: split_ltm_virtual_address_status_or_irule_profile_err.__func__, 17242599: split_ltm_virtual_address_status_or_irule_profile_err.__func__, 17242600: split_ltm_virtual_address_status_or_irule_profile_err.__func__, 17242602: split_gtm_changed_state.__func__, 17243049: split_ltm_virtual_status.__func__, 18415620: split_tmm_address_conflict.__func__, 18485252: split_gtm_monitor.__func__, 18485253: split_gtm_monitor.__func__, 18485505: split_gtm_monitor.__func__, 18485506: split_gtm_monitor.__func__, 18493443: split_gtm_monitor.__func__, 18493444: split_gtm_monitor.__func__, 18497538: split_gtm_monitor.__func__, 18497539: split_gtm_monitor.__func__, 18497540: split_gtm_monitor.__func__, 18497793: split_gtm_monitor.__func__, 18497794: split_gtm_monitor.__func__, 18501635: split_gtm_monitor.__func__, 18501636: split_gtm_monitor.__func__, 18501640: split_gtm_syncgroup_change.__func__, 18501641: split_gtm_syncgroup_change.__func__, 18501643: split_gtm_monitor.__func__, 18501644: split_gtm_monitor.__func__, 18505733: split_gtm_monitor.__func__, 18505734: split_gtm_monitor.__func__, 18526211: split_gtm_monitor.__func__, 18526212: split_gtm_monitor.__func__, 18538738: split_gtm_monitor_instance.__func__, 18743298: split_ltm_sweeper_active2.__func__, 18743299: split_ltm_sweeper_active3.__func__, 18808837: split_ltm_http_header_exceeded.__func__, 18808849: split_ltm_http_header_exceeded.__func__, 18808839: split_ltm_http_process_state.__func__, 18808854: split_ltm_http_process_state.__func__, 19005441: split_ltm_rule.__func__, 19005442: split_ltm_rule.__func__, 19005447: split_ltm_rule.__func__, 19005448: split_ltm_rule.__func__, 19005449: split_ltm_rule.__func__, 19005456: split_ltm_rule.__func__, 19005457: split_ltm_rule.__func__, 18874386: split_ltm_conn_limit_reached.__func__, 18874388: split_ltm_conn_limit_reached.__func__, 19071296: split_ltm_rst_reason.__func__, 19267603: split_ltm_ssl_handshake_fail.__func__, 19267622: split_ltm_shared_ciphers.__func__, 19267592: split_ltm_connection_error.__func__, 19267593: split_ltm_connection_error.__func__, 21102594: split_audit_bigip_tmsh_audit.__func__, 21102599: split_ltm_cert_expiry.__func__, 21102600: split_ltm_cert_expiry.__func__, 21954565: split_audit_scriptd_run_script.__func__, 22220814: split_ltm_dns_failed_xfr_rcode.__func__, 22220821: split_ltm_dns_failed_rr.__func__, 22220824: split_ltm_dns_failed_xfr.__func__, 22220828: split_ltm_dns_handling_notify.__func__, 22220831: split_ltm_dns_axfr_succeeded_1f.__func__, 22220834: split_ltm_dns_ignoring_tfer.__func__, 22220844: split_ltm_dns_axfr_succeeded_2c.__func__}
    no_split_logs = ('Per-invocation log rate exceeded; throttling', 'Resuming log processing at this invocation', 'Re-enabling general logging;', 'Cumulative log rate exceeded!  Throttling all non-debug logs.')
    extra_cols = {'rawmsg', 'timestamp', 'host', 'level', 'process', 'proc_pid', 'logid1', 'logid2', 'message', 'object'}
    colorizers = [CellColorizer(100, None, colorizeMonitors), RowColorizer(101, None, colorizeRows)]

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        try:
            import zoneinfo
        except ImportError:
            from backports import zoneinfo
        self._log_tz = zoneinfo.ZoneInfo('UTC')
        try:
            self._year = int(vd.options.get('f5log_log_year', datetime.utcfromtimestamp(self.source.stat().st_ctime).year))
        except (AttributeError, ValueError, TypeError):
            self._year = datetime.now().year

    def iterload(self):
        if False:
            i = 10
            return i + 15
        self.rows = []
        if vd.options.get('f5log_object_regex'):
            try:
                object_regex = re.compile(vd.options.get('f5log_object_regex'))
            except re.error as exc:
                object_regex = None
        else:
            object_regex = None
        try:
            self._log_tz = zoneinfo.ZoneInfo(vd.options.get('f5log_log_timzeone', 'UTC'))
        except zoneinfo.ZoneInfoNotFoundError as exc:
            self._log_tz = zoneinfo.ZoneInfo('UTC')
        for line in self.source:
            m = F5LogSheet.re_f5log.match(line)
            if m:
                m = m.groupdict()
            else:
                yield F5LogSheet.F5LogRow(rawmsg=line, kv={'PARSE_ERROR': 'unable to parse line'})
                continue
            kv = {'message': m.get('message')}
            if m.get('date1'):
                _t = m.get('date1')
                try:
                    timestamp = datetime(year=self._year, month=self._months[_t[:3]], day=int(_t[4:6]), hour=int(_t[7:9]), minute=int(_t[10:12]), second=int(_t[13:15]), tzinfo=self._log_tz)
                except ValueError as exc:
                    yield F5LogSheet.F5LogRow(rawmsg=line, PARSE_ERROR='\n'.join(traceback.format_exception(etype=type(exc), value=exc, tb=exc.__traceback__)))
            elif m.get('date2'):
                timestamp = datetime.strptime(m.get('date2'), '%Y-%m-%dT%H:%M:%S%z')
            elif m.get('date3'):
                timestamp = datetime.strptime(f"{self._year}-{m.get('date3')}", '%Y-%m-%d %H:%M:%S')
                timestamp = datetime(year=timestamp.year, month=timestamp.month, day=timestamp.day, hour=timestamp.hour, minute=timestamp.minute, second=timestamp.second, tzinfo=self._log_tz)
                (m['level'], m['host']) = (m.get('host'), m.get('level'))
            else:
                timestamp = None
            logid1 = int(m.get('logid1'), base=16) if m.get('logid1') else None
            if logid1 in self.splitters and (not any((m.get('message', '').startswith(_) for _ in F5LogSheet.no_split_logs))):
                try:
                    for entry in F5LogSheet.splitters[logid1](m.get('message')):
                        kv.update(entry)
                except (IndexError, ValueError) as exc:
                    yield F5LogSheet.F5LogRow(rawmsg=line, PARSE_ERROR='\n'.join(traceback.format_exception(etype=type(exc), value=exc, tb=exc.__traceback__)))
                if 'object' in kv and object_regex:
                    om = object_regex.match(kv.get('object', ''))
                    if om:
                        kv.update(om.groupdict())
                for (k, v) in kv.items():
                    if k not in self.extra_cols:
                        F5LogSheet.addColumn(self, AttrColumn(k))
                        self.extra_cols.add(k)
            elif logid1 is None and m.get('message').startswith('Rule '):
                for entry in self.split_ltm_rule(m.get('message')):
                    kv.update(entry)
            yield F5LogSheet.F5LogRow(timestamp=timestamp, host=m.get('host'), level=m.get('level'), process=m.get('process'), proc_pid=int(m.get('pid')) if m.get('pid') is not None else None, logid1=m.get('logid1') if m.get('logid1') is not None else None, logid2=m.get('logid2') if m.get('logid2') is not None else None, **kv)

@VisiData.api
def open_f5log(vd: VisiData, p: Path) -> Sheet:
    if False:
        i = 10
        return i + 15
    sheet = F5LogSheet(p.name, source=p)
    sheet.options['disp_date_fmt'] = '%Y-%m-%d %H:%M:%S'
    return sheet