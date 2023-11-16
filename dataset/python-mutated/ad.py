from ldap3.protocol.formatters.formatters import format_sid
from impacket.ldap.ldaptypes import SR_SECURITY_DESCRIPTOR
from pupylib.PupyConfig import PupyConfig
from pupylib.PupyModule import config, PupyModule, PupyArgumentParser
from pupylib.PupyOutput import Pygment, List, Table, MultiPart
from pygments import lexers
from os.path import basename
from json import dumps
from io import open as io_open
from threading import Event
from datetime import datetime
from uuid import UUID
from collections import OrderedDict
__class_name__ = 'AD'
IMM = 0
LIST = 1
MAP = 2
DATE = 3
ALL_ATTRIBUTES = '*'
NO_ATTRIBUTES = '1.1'
ALL_OPERATIONAL_ATTRIBUTES = '+'
BASE = 'BASE'
LEVEL = 'LEVEL'
SUBTREE = 'SUBTREE'
uac_flags = {'ACCOUNT_DISABLED': 2, 'ACCOUNT_LOCKED': 16, 'PASSWD_NOTREQD': 32, 'PASSWD_CANT_CHANGE': 64, 'NORMAL_ACCOUNT': 512, 'WORKSTATION_ACCOUNT': 4096, 'SERVER_TRUST_ACCOUNT': 8192, 'DONT_EXPIRE_PASSWD': 65536, 'SMARTCARD_REQUIRED': 262144, 'TRUSTED_FOR_DELEGATION': 524288, 'NOT_DELEGATED': 1048576, 'USE_DES_KEY_ONLY': 2097152, 'DONT_REQ_PREAUTH': 4194304, 'PASSWORD_EXPIRED': 8388608, 'TRUSTED_TO_AUTH_FOR_DELEGATION': 16777216, 'PARTIAL_SECRETS_ACCOUNT': 67108864}
sam_flags = {'SAM_DOMAIN_OBJECT': 0, 'SAM_GROUP_OBJECT': 268435456, 'SAM_NON_SECURITY_GROUP_OBJECT': 268435457, 'SAM_ALIAS_OBJECT': 536870912, 'SAM_NON_SECURITY_ALIAS_OBJECT': 536870913, 'SAM_NORMAL_USER_ACCOUNT': 805306368, 'SAM_MACHINE_ACCOUNT': 805306369, 'SAM_TRUST_ACCOUNT': 805306370, 'SAM_APP_BASIC_GROUP': 1073741824, 'SAM_APP_QUERY_GROUP': 1073741825, 'SAM_ACCOUNT_TYPE_MAX': 2147483647}
pwd_flags = {'PASSWORD_COMPLEX': 1, 'PASSWORD_NO_ANON_CHANGE': 2, 'PASSWORD_NO_CLEAR_CHANGE': 4, 'LOCKOUT_ADMINS': 8, 'PASSWORD_STORE_CLEARTEXT': 16, 'REFUSE_PASSWORD_CHANGE': 32}
trust_flags = {'NON_TRANSITIVE': 1, 'UPLEVEL_ONLY': 2, 'QUARANTINED_DOMAIN': 4, 'FOREST_TRANSITIVE': 8, 'CROSS_ORGANIZATION': 16, 'WITHIN_FOREST': 32, 'TREAT_AS_EXTERNAL': 64, 'USES_RC4_ENCRYPTION': 128, 'CROSS_ORGANIZATION_NO_TGT_DELEGATION': 512, 'PIM_TRUST': 1024}
trust_directions = {'INBOUND': 1, 'OUTBOUND': 2, 'BIDIRECTIONAL': 3}
trust_type = {'DOWNLEVEL': 1, 'UPLEVEL': 2, 'MIT': 3}
attr_translations = {'sAMAccountName': 'SAM Name', 'cn': 'CN', 'operatingSystem': 'Operating System', 'operatingSystemServicePack': 'Service Pack', 'operatingSystemVersion': 'OS Version', 'userAccountControl': 'Flags', 'objectSid': 'SID', 'memberOf': 'Member of groups', 'primaryGroupId': 'Primary group', 'dNSHostName': 'DNS Hostname', 'whenCreated': 'Created on', 'whenChanged': 'Changed on', 'IPv4': 'IPv4 Address', 'lockOutObservationWindow': 'Lockout time window', 'lockoutDuration': 'Lockout Duration', 'lockoutThreshold': 'Lockout Threshold', 'maxPwdAge': 'Max password age', 'minPwdAge': 'Min password age', 'minPwdLength': 'Min password length', 'pwdHistoryLength': 'Password history length', 'pwdProperties': 'Password properties', 'ms-DS-MachineAccountQuota': 'Machine Account Quota', 'flatName': 'NETBIOS Domain name'}
sid_translations = {'S-1-0': ('Null Authority', 'USER'), 'S-1-0-0': ('Nobody', 'USER'), 'S-1-1': ('World Authority', 'USER'), 'S-1-1-0': ('Everyone', 'GROUP'), 'S-1-2': ('Local Authority', 'USER'), 'S-1-2-0': ('Local', 'GROUP'), 'S-1-2-1': ('Console Logon', 'GROUP'), 'S-1-3': ('Creator Authority', 'USER'), 'S-1-3-0': ('Creator Owner', 'USER'), 'S-1-3-1': ('Creator Group', 'GROUP'), 'S-1-3-2': ('Creator Owner Server', 'COMPUTER'), 'S-1-3-3': ('Creator Group Server', 'COMPUTER'), 'S-1-3-4': ('Owner Rights', 'GROUP'), 'S-1-4': ('Non-unique Authority', 'USER'), 'S-1-5': ('NT Authority', 'USER'), 'S-1-5-1': ('Dialup', 'GROUP'), 'S-1-5-2': ('Network', 'GROUP'), 'S-1-5-3': ('Batch', 'GROUP'), 'S-1-5-4': ('Interactive', 'GROUP'), 'S-1-5-6': ('Service', 'GROUP'), 'S-1-5-7': ('Anonymous', 'GROUP'), 'S-1-5-8': ('Proxy', 'GROUP'), 'S-1-5-9': ('Enterprise Domain Controllers', 'GROUP'), 'S-1-5-10': ('Principal Self', 'USER'), 'S-1-5-11': ('Authenticated Users', 'GROUP'), 'S-1-5-12': ('Restricted Code', 'GROUP'), 'S-1-5-13': ('Terminal Server Users', 'GROUP'), 'S-1-5-14': ('Remote Interactive Logon', 'GROUP'), 'S-1-5-15': ('This Organization ', 'GROUP'), 'S-1-5-17': ('This Organization ', 'GROUP'), 'S-1-5-18': ('Local System', 'USER'), 'S-1-5-19': ('NT Authority', 'USER'), 'S-1-5-20': ('NT Authority', 'USER'), 'S-1-5-80-0': ('All Services ', 'GROUP'), 'S-1-5-32-544': ('Administrators', 'GROUP'), 'S-1-5-32-545': ('Users', 'GROUP'), 'S-1-5-32-546': ('Guests', 'GROUP'), 'S-1-5-32-547': ('Power Users', 'GROUP'), 'S-1-5-32-548': ('Account Operators', 'GROUP'), 'S-1-5-32-549': ('Server Operators', 'GROUP'), 'S-1-5-32-550': ('Print Operators', 'GROUP'), 'S-1-5-32-551': ('Backup Operators', 'GROUP'), 'S-1-5-32-552': ('Replicators', 'GROUP'), 'S-1-5-32-554': ('Pre-Windows 2000 Compatible Access', 'GROUP'), 'S-1-5-32-555': ('Remote Desktop Users', 'GROUP'), 'S-1-5-32-556': ('Network Configuration Operators', 'GROUP'), 'S-1-5-32-557': ('Incoming Forest Trust Builders', 'GROUP'), 'S-1-5-32-558': ('Performance Monitor Users', 'GROUP'), 'S-1-5-32-559': ('Performance Log Users', 'GROUP'), 'S-1-5-32-560': ('Windows Authorization Access Group', 'GROUP'), 'S-1-5-32-561': ('Terminal Server License Servers', 'GROUP'), 'S-1-5-32-562': ('Distributed COM Users', 'GROUP'), 'S-1-5-32-568': ('IIS_IUSRS', 'GROUP'), 'S-1-5-32-569': ('Cryptographic Operators', 'GROUP'), 'S-1-5-32-573': ('Event Log Readers', 'GROUP'), 'S-1-5-32-574': ('Certificate Service DCOM Access', 'GROUP'), 'S-1-5-32-575': ('RDS Remote Access Servers', 'GROUP'), 'S-1-5-32-576': ('RDS Endpoint Servers', 'GROUP'), 'S-1-5-32-577': ('RDS Management Servers', 'GROUP'), 'S-1-5-32-578': ('Hyper-V Administrators', 'GROUP'), 'S-1-5-32-579': ('Access Control Assistance Operators', 'GROUP'), 'S-1-5-32-580': ('Access Control Assistance Operators', 'GROUP')}
access_mask_flags = OrderedDict([('GenericAll', 983551), ('GenericWrite', 131112), ('GenericRead', 131220), ('GENERIC_EXECUTE', 131076), ('ACCESS_SYSTEM_SECURITY', 16777216), ('SYNCHRONIZE', 1048576), ('WriteOwner', 524288), ('WriteDacl', 262144), ('READ_CONTROL', 131072), ('DELETE', 65536), ('ADS_RIGHT_DS_CONTROL_ACCESS', 256), ('ADS_RIGHT_DS_CREATE_CHILD', 1), ('ADS_RIGHT_DS_DELETE_CHILD', 2), ('ADS_RIGHT_DS_READ_PROP', 16), ('ADS_RIGHT_DS_WRITE_PROP', 32), ('ADS_RIGHT_DS_SELF', 8)])

def _sid(sid):
    if False:
        i = 10
        return i + 15
    (value, _) = sid_translations.get(sid, (None, None))
    if value:
        return value
    return sid

def json_default(o):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(o, datetime):
        return o.isoformat()

def nsToDays(length):
    if False:
        while True:
            i = 10
    return abs(length) * 1e-07 / 86400

def nsToMinutes(length):
    if False:
        return 10
    return abs(length) * 1e-07 / 60

def toDateTime(filetime):
    if False:
        return 10
    if isinstance(filetime, datetime):
        return filetime
    if not filetime:
        return datetime.utcfromtimestamp(0)
    return datetime.utcfromtimestamp(float(filetime / 10000000 - 11644473600))

def parseFlags(attr, flags_def, bits=True):
    if False:
        i = 10
        return i + 15
    if not attr:
        return tuple()
    if not isinstance(attr, int):
        attr = int(attr)
    return tuple((flag for (flag, val) in flags_def.iteritems() if bits and attr & val == val or (not bits and attr == val)))

def LDAPAclMaskToSet(mask):
    if False:
        for i in range(10):
            print('nop')
    result = []
    rest = mask['Mask']
    for (flag, value) in access_mask_flags.iteritems():
        if rest & value == value:
            result.append(flag)
            rest &= ~value
            if not rest:
                break
    if not result and rest:
        result.append(rest)
    return result

def LDAPAclToDict(acl):
    if False:
        while True:
            i = 10
    if not acl:
        return None
    result = []
    for ace in acl.aces:
        sid = ace['Ace']['Sid'].formatCanonical()
        result.append({'Type': ace['TypeName'][:-4], 'Sid': _sid(sid), 'Mask': LDAPAclMaskToSet(ace['Ace']['Mask'])})
    return result

def LDAPAclOwnerToDict(owner):
    if False:
        print('Hello World!')
    if not owner:
        return None
    sid = owner.formatCanonical()
    return _sid(sid)

def LDAPSdToDict(descriptor):
    if False:
        return 10
    if not descriptor:
        return None
    return {'Owner': LDAPAclOwnerToDict(descriptor['OwnerSid']), 'Group': LDAPAclOwnerToDict(descriptor['GroupSid']), 'SACL': LDAPAclToDict(descriptor['Sacl']), 'DACL': LDAPAclToDict(descriptor['Dacl'])}

def formatAttribute(key, att, formatCnAsGroup=False):
    if False:
        while True:
            i = 10
    aname = key.lower()
    if isinstance(att, tuple) and len(att) == 1 and (not isinstance(att[0], dict)):
        att = att[0]
        if isinstance(att, (str, unicode)):
            att = att.strip()
            try:
                att = int(att)
            except ValueError:
                try:
                    att = float(att)
                except ValueError:
                    pass
    if aname == 'useraccountcontrol':
        return parseFlags(att, uac_flags)
    elif aname == 'pwdproperties':
        return parseFlags(att, pwd_flags)
    elif aname == 'samaccounttype':
        return parseFlags(att, sam_flags, False)
    elif aname == 'trustattributes':
        return parseFlags(att, trust_flags)
    elif aname == 'trustdirection':
        if att == 0:
            return 'DISABLED'
        else:
            return parseFlags(att, trust_directions, False)
    elif aname == 'trusttype':
        return parseFlags(att, trust_type)
    elif aname in ('securityidentifier', 'objectsid') and att.startswith('hex:'):
        sid = format_sid(att[4:].decode('hex'))
        return _sid(sid)
    elif aname == 'minpwdage' or aname == 'maxpwdage':
        return '%.2f days' % nsToDays(att)
    elif aname == 'lockoutobservationwindow' or aname == 'lockoutduration':
        return '%.1f minutes' % nsToMinutes(att)
    elif aname == 'objectguid' and att.startswith('hex:'):
        return '{' + str(UUID(att[4:])) + '}'
    elif aname in ('pwdlastchange', 'badpasswordtime', 'lastlogon', 'lastlogontimestamp', 'lockouttime'):
        return toDateTime(att)
    elif aname in ('ntsecuritydescriptor',):
        if att.startswith('hex:'):
            att = att[4:].decode('hex')
        srsd = SR_SECURITY_DESCRIPTOR()
        srsd.fromString(att)
        return LDAPSdToDict(srsd)
    return att

def from_tuple_deep(obj, format=True):
    if False:
        i = 10
        return i + 15
    (kind, data) = obj
    if kind == IMM:
        if isinstance(data, str):
            try:
                return data.decode('utf-8')
            except UnicodeDecodeError:
                return 'hex:' + data.encode('hex')
        return data
    elif kind == LIST:
        return tuple((from_tuple_deep(item) for item in data))
    elif kind == MAP and format:
        return {k: formatAttribute(k, from_tuple_deep(v)) for (k, v) in data}
    elif kind == MAP and (not format):
        return {k: from_tuple_deep(v) for (k, v) in data}
    elif kind == DATE:
        return datetime.utcfromtimestamp(data)
    else:
        raise ValueError('Invalid kind ({})'.format(kind))

def _get_field(result, field):
    if False:
        i = 10
        return i + 15
    if field in result:
        return result[field]
    l_field = field.lower()
    for r_field in result:
        if r_field.lower() == l_field:
            return result[r_field]
    raise AttributeError(field)

@config(cat='admin')
class AD(PupyModule):
    """ Dump information from Active Directory """
    dependencies = {'all': ['gssapi', 'ldap3'], 'posix': ['kerberos'], 'windows': ['winkerberos']}
    terminate = None

    @classmethod
    def init_argparse(cls):
        if False:
            i = 10
            return i + 15
        cls.arg_parser = PupyArgumentParser(prog='ad', description=cls.__doc__)
        cls.arg_parser.add_argument('-G', '--global-catalog', default=False, action='store_true', help='Use AD Global catalg')
        cls.arg_parser.add_argument('-r', '--realm', help='Realm to work with')
        commands = cls.arg_parser.add_subparsers(title='commands')
        bind = commands.add_parser('bind', help='Bind to server')
        bind.add_argument('-l', '--ldap-server', help='DNS address of LDAP server')
        bind.add_argument('-T', '--recv-timeout', default=60, help='Socket read timeout')
        bind.add_argument('-u', '--username', help='Username to authenticate')
        bind.add_argument('-p', '--password', help='Password to authenticate')
        bind.add_argument('-d', '--domain', help='Domain for Username')
        bind.add_argument('-r', '--root', help='LDAP root')
        bind.set_defaults(func=cls.bind)
        unbind = commands.add_parser('unbind', help='Disconnect and forget realm')
        unbind.set_defaults(func=cls.unbind)
        bounded = commands.add_parser('list', help='Show bounded realms')
        bounded.set_defaults(func=cls.bounded)
        info = commands.add_parser('info', help='Info about current AD context')
        info.set_defaults(func=cls.getinfo)
        dump = commands.add_parser('dump', help='Dump results of large searches')
        dump.add_argument('-f', '--filter', help='LDAP custom filter')
        dump.add_argument('-F', '--full', default=False, action='store_true', help='Dump all attributes')
        dump.add_argument('target', nargs='?', help='Categories to dump, i.e.: users,computers OR filter like (&(attr=XYZ)(attr2=CCC))')
        dump.set_defaults(func=cls.dump)
        childs = commands.add_parser('childs', help='Related AD servers')
        childs.set_defaults(func=cls.childs)
        search = commands.add_parser('search', help='Search in AD (only small and fast, for large use dump)')
        search.add_argument('-T', '--table', action='store_true', default=False, help='Output as table')
        search.add_argument('term', help='Search filter', default='(objectClass=domain)')
        search.add_argument('attributes', nargs='?', help='Attributes to search (Use * for ALL, default none)')
        level = search.add_mutually_exclusive_group()
        level.add_argument('-1', '--base', default=False, action='store_true', help='Use base search instead of subtree search. Default: False')
        level.add_argument('-L', '--level', default=False, action='store_true', help='Use level search instead of subtree search. Default: False')
        search.add_argument('-r', '--root', help='Use root instead of autodiscovered one')
        search.add_argument('-n', '--amount', default=5, type=int, help='Amount of records. Default: 5')
        search.add_argument('-t', '--timeout', default=5, type=int, help='Timeout (seconds). Default: 5')
        search.set_defaults(func=cls.search)

    def _show_exception(self, e):
        if False:
            return 10
        if hasattr(e, 'message') and hasattr(e, 'type'):
            report = []
            if hasattr(e, 'childs') and e.childs and (not isinstance(e.childs, str)):
                for (authentication, ldap_server, domain, user, _, emessage) in e.childs:
                    report.append({'Method': authentication, 'Server': ldap_server, 'Domain': domain, 'User': user, 'Message': emessage})
                self.error(Table(report, ['Method', 'Server', 'Domain', 'User', 'Message'], caption=e.message))
            else:
                self.error('AD Error ({}): {}'.format(e.type, e.message))
        else:
            self.error(e)

    def run(self, args):
        if False:
            print('Hello World!')
        try:
            if args.realm == '.':
                args.realm = None
            args.func(self, args)
        except Exception as e:
            self._show_exception(e)

    def _output_search_results(self, results, fields, table=False, realm=None):
        if False:
            print('Hello World!')
        if not results:
            return
        is_list = False
        is_table = False
        if len(fields) == 1:
            _results = [_get_field(line, fields[0]) for line in results]
            is_list = all((not isinstance(record, (dict, tuple, list)) for record in _results))
            if is_list:
                results = _results
        elif table and fields:
            results = [{field: _get_field(result, field) for field in fields} for result in results]
            is_table = all((all((not isinstance(value, (dict, tuple, list)) for value in record.itervalues())) for record in results))
        if is_list:
            self.log(List(results, caption=realm))
        elif is_table:
            self.log(Table(results, fields or None, caption=realm))
        else:
            filtered = results
            if fields:
                filtered = [{field: _get_field(result, field) for field in fields} for result in results]
            formatted_json = dumps(filtered, indent=2, sort_keys=True, default=json_default, ensure_ascii=False)
            if realm:
                self.log('+ ' + realm)
            self.log(Pygment(lexers.JsonLexer(), formatted_json))

    def search(self, args):
        if False:
            print('Hello World!')
        search = self.client.remote('ad', 'search')
        level = SUBTREE
        if args.base:
            level = BASE
        elif args.level:
            level = LEVEL
        need_attrs = []
        term = args.term
        attributes = args.attributes
        if term:
            term = term.strip()
        if attributes:
            attributes = attributes.strip()
        if not attributes:
            if term and (not term.startswith('(')):
                attributes = term
                term = '(objectClass=*)'
            else:
                attributes = NO_ATTRIBUTES
        fields = []
        if attributes in (ALL_ATTRIBUTES, ALL_OPERATIONAL_ATTRIBUTES):
            pass
        elif attributes == NO_ATTRIBUTES:
            fields.append('dn')
        else:
            attributes = [attribute.strip() for attribute in attributes.split(',')]
            for attribute in attributes:
                fields.append(attribute)
                if attribute.lower() == 'dn':
                    continue
                need_attrs.append(attribute)
            attributes = tuple(need_attrs)
        (ok, result) = search(args.realm, args.global_catalog, term, attributes, level, args.root, args.amount, args.timeout, False)
        if not ok:
            self.error(result)
            return
        results = from_tuple_deep(result, True)
        if isinstance(results, dict):
            for (realm, results) in results.iteritems():
                self._output_search_results(results, fields, args.table, realm)
        else:
            self._output_search_results(results, fields, args.table)

    def unbind(self, args):
        if False:
            for i in range(10):
                print('nop')
        unbind = self.client.remote('ad', 'unbind')
        unbind(args.realm, args.global_catalog)

    def childs(self, args):
        if False:
            while True:
                i = 10
        childs = self.client.remote('ad', 'childs')
        (ok, result) = childs(args.realm, args.global_catalog)
        if not ok:
            self.error(result)
            return
        (i_am, rootdn, childs) = result
        self.log(List(childs, caption='Root: {} Whoami: {}'.format(rootdn, i_am)))

    def bounded(self, args):
        if False:
            return 10
        bounded = self.client.remote('ad', 'bounded')
        self.log(Table([{'TYPE': btype, 'REALM': realm} for (btype, realm) in bounded()], ['TYPE', 'REALM']))

    def getinfo(self, args):
        if False:
            return 10
        info = self.client.remote('ad', 'info')
        desc = from_tuple_deep(info(args.realm, args.global_catalog), False)
        idesc = desc.get('info', {})
        infos = []
        versions = idesc.get('supported_ldap_versions', [])
        if not hasattr(versions, '__iter__'):
            versions = [versions]
        infos.append(List(['Bind: ' + desc.get('bind', ''), 'Root: ' + desc.get('root', ''), 'LDAP: ' + desc.get('ldap', ''), 'DNS: ' + desc['dns'][4][0] if desc.get('dns', None) else '', 'Schema: ' + idesc.get('schema_entry', ''), 'Versions: ' + ', '.join((str(version) for version in versions)), 'SASL Mechs: ' + ', '.join((mech for mech in idesc.get('supported_sasl_mechanisms', [])))], caption='Connection'))
        if desc['ldap_servers']:
            infos.append(Table(desc['ldap_servers'], ['address', 'port', 'priority'], caption='LDAP Servers'))
        if desc['dns_servers']:
            infos.append(Table([{'IP': dns[0][4][0] + ('/tcp' if dns[0][2] == 2 else '/udp'), 'Delay': '{:.02f}ms'.format(dns[1] * 1000)} for dns in desc['dns_servers']], ['IP', 'Delay'], caption='DNS Servers'))
        if not idesc:
            self.log(MultiPart(infos))
            return
        if idesc['alt_servers']:
            infos.append(List(idesc['alt_servers'], caption='Alternate servers'))
        if idesc['naming_contexts'] and (not isinstance(idesc['naming_contexts'], (str, unicode))):
            infos.append(List(idesc['naming_contexts'], caption='Naming contexts'))
        supported = []
        for table in ('supported_controls', 'supported_extensions', 'supported_features'):
            for (oid, klass, name, vendor) in idesc[table]:
                supported.append({'OID': oid, 'Type': klass, 'Name': name, 'Vendor': vendor})
        if supported:
            infos.append(Table(supported, ['OID', 'Type', 'Name', 'Vendor'], caption='Supported features and extensions'))
        if 'other' in idesc:
            infos.append(List(tuple(('{}: {}'.format(key, value) for (key, value) in idesc['other'].iteritems() if key not in ('supportedLDAPPolicies',))), caption='Other info'))
        self.log(MultiPart(infos))

    def dump(self, args):
        if False:
            while True:
                i = 10
        addump = self.client.remote('ad', 'dump', False)
        config = self.client.pupsrv.config or PupyConfig()
        context = {'last_name': None, 'last_fobj': None}
        completion = Event()

        def on_data(name, payload):
            if False:
                for i in range(10):
                    print('nop')
            if name == 'dns':
                self.success('DNS server selected: {}'.format(payload))
            elif name == 'ldap':
                self.success('LDAP server selected: {}'.format(payload))
            elif name == 'error':
                self.error('Error: {}'.format(payload))
            else:
                is_first = False
                if context['last_name'] != name:
                    is_first = True
                    if context['last_name']:
                        self.success('Completed {}'.format(context['last_name']))
                    dest = config.get_file('ad', {'%c': self.client.short_name(), '%n': basename(name), '%r': args.realm})
                    self.success('Dumping {} -> {}'.format(name, dest))
                    context['last_name'] = name
                    if context['last_fobj']:
                        context['last_fobj'].write(u'\n]')
                        context['last_fobj'].close()
                    context['last_fobj'] = io_open(dest, 'w+', encoding='utf-8')
                    context['last_fobj'].write(u'[\n')
                for record in from_tuple_deep(payload):
                    if is_first:
                        is_first = False
                    else:
                        context['last_fobj'].write(u',')
                    record_json = dumps(record, indent=2, sort_keys=True, default=json_default, ensure_ascii=False)
                    context['last_fobj'].write(record_json)

        def on_complete():
            if False:
                while True:
                    i = 10
            if context['last_fobj']:
                context['last_fobj'].write(u']')
                context['last_fobj'].close()
                context['last_fobj'] = None
            if context['last_name']:
                self.success('Completed {}'.format(context['last_name']))
                context['last_name'] = None
                self.success('Dump completed')
            completion.set()
        self.terminate = addump(on_data, on_complete, args.realm, args.global_catalog, args.filter or args.target, not args.full)
        completion.wait()

    def bind(self, args):
        if False:
            i = 10
            return i + 15
        bind = self.client.remote('ad', 'bind', False)
        completion = Event()

        def on_data(payload):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(payload, tuple):
                self.info(List(payload[1], caption=payload[0]))
            else:
                self.info(payload)

        def on_completed(success, payload):
            if False:
                i = 10
                return i + 15
            try:
                if success:
                    self.success('Bound to server: {}'.format(payload))
                else:
                    self._show_exception(payload)
            finally:
                completion.set()
        self.terminate = bind(on_data, on_completed, args.realm, args.global_catalog, args.ldap_server, args.recv_timeout, args.domain, args.username, args.password, args.root)
        completion.wait()

    def interrupt(self):
        if False:
            for i in range(10):
                print('nop')
        if self.terminate:
            self.terminate()
        else:
            raise NotImplementedError()