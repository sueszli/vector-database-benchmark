"""
DRBD administration module
"""
import logging
log = logging.getLogger(__name__)

def _analyse_overview_field(content):
    if False:
        for i in range(10):
            print('nop')
    '\n    Split the field in drbd-overview\n    '
    if '(' in content:
        return (content.split('(')[0], content.split('(')[0])
    elif '/' in content:
        return (content.split('/')[0], content.split('/')[1])
    return (content, '')

def _count_spaces_startswith(line):
    if False:
        print('Hello World!')
    '\n    Count the number of spaces before the first character\n    '
    if line.split('#')[0].strip() == '':
        return None
    spaces = 0
    for i in line:
        if i.isspace():
            spaces += 1
        else:
            return spaces

def _analyse_status_type(line):
    if False:
        while True:
            i = 10
    '\n    Figure out the sections in drbdadm status\n    '
    spaces = _count_spaces_startswith(line)
    if spaces is None:
        return ''
    switch = {0: 'RESOURCE', 2: {' disk:': 'LOCALDISK', ' role:': 'PEERNODE', ' connection:': 'PEERNODE'}, 4: {' peer-disk:': 'PEERDISK'}}
    ret = switch.get(spaces, 'UNKNOWN')
    if isinstance(ret, str):
        return ret
    for x in ret:
        if x in line:
            return ret[x]
    return 'UNKNOWN'

def _add_res(line):
    if False:
        for i in range(10):
            print('nop')
    '\n    Analyse the line of local resource of ``drbdadm status``\n    '
    global resource
    fields = line.strip().split()
    if resource:
        ret.append(resource)
        resource = {}
    resource['resource name'] = fields[0]
    resource['local role'] = fields[1].split(':')[1]
    resource['local volumes'] = []
    resource['peer nodes'] = []

def _add_volume(line):
    if False:
        i = 10
        return i + 15
    '\n    Analyse the line of volumes of ``drbdadm status``\n    '
    section = _analyse_status_type(line)
    fields = line.strip().split()
    volume = {}
    for field in fields:
        volume[field.split(':')[0]] = field.split(':')[1]
    if section == 'LOCALDISK':
        resource['local volumes'].append(volume)
    else:
        lastpnodevolumes.append(volume)

def _add_peernode(line):
    if False:
        for i in range(10):
            print('nop')
    '\n    Analyse the line of peer nodes of ``drbdadm status``\n    '
    global lastpnodevolumes
    fields = line.strip().split()
    peernode = {}
    peernode['peernode name'] = fields[0]
    peernode[fields[1].split(':')[0]] = fields[1].split(':')[1]
    peernode['peer volumes'] = []
    resource['peer nodes'].append(peernode)
    lastpnodevolumes = peernode['peer volumes']

def _empty(dummy):
    if False:
        return 10
    '\n    Action of empty line of ``drbdadm status``\n    '

def _unknown_parser(line):
    if False:
        i = 10
        return i + 15
    '\n    Action of unsupported line of ``drbdadm status``\n    '
    global ret
    ret = {'Unknown parser': line}

def _line_parser(line):
    if False:
        for i in range(10):
            print('nop')
    '\n    Call action for different lines\n    '
    section = _analyse_status_type(line)
    fields = line.strip().split()
    switch = {'': _empty, 'RESOURCE': _add_res, 'PEERNODE': _add_peernode, 'LOCALDISK': _add_volume, 'PEERDISK': _add_volume}
    func = switch.get(section, _unknown_parser)
    func(line)

def overview():
    if False:
        i = 10
        return i + 15
    "\n    Show status of the DRBD devices, support two nodes only.\n    drbd-overview is removed since drbd-utils-9.6.0,\n    use status instead.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' drbd.overview\n    "
    cmd = 'drbd-overview'
    for line in __salt__['cmd.run'](cmd).splitlines():
        ret = {}
        fields = line.strip().split()
        minnum = fields[0].split(':')[0]
        device = fields[0].split(':')[1]
        (connstate, _) = _analyse_overview_field(fields[1])
        (localrole, partnerrole) = _analyse_overview_field(fields[2])
        (localdiskstate, partnerdiskstate) = _analyse_overview_field(fields[3])
        if localdiskstate.startswith('UpTo'):
            if partnerdiskstate.startswith('UpTo'):
                if len(fields) >= 5:
                    mountpoint = fields[4]
                    fs_mounted = fields[5]
                    totalsize = fields[6]
                    usedsize = fields[7]
                    remainsize = fields[8]
                    perc = fields[9]
                    ret = {'minor number': minnum, 'device': device, 'connection state': connstate, 'local role': localrole, 'partner role': partnerrole, 'local disk state': localdiskstate, 'partner disk state': partnerdiskstate, 'mountpoint': mountpoint, 'fs': fs_mounted, 'total size': totalsize, 'used': usedsize, 'remains': remainsize, 'percent': perc}
                else:
                    ret = {'minor number': minnum, 'device': device, 'connection state': connstate, 'local role': localrole, 'partner role': partnerrole, 'local disk state': localdiskstate, 'partner disk state': partnerdiskstate}
            else:
                syncbar = fields[4]
                synced = fields[6]
                syncedbytes = fields[7]
                sync = synced + syncedbytes
                ret = {'minor number': minnum, 'device': device, 'connection state': connstate, 'local role': localrole, 'partner role': partnerrole, 'local disk state': localdiskstate, 'partner disk state': partnerdiskstate, 'synchronisation: ': syncbar, 'synched': sync}
    return ret
ret = []
resource = {}
lastpnodevolumes = None

def status(name='all'):
    if False:
        while True:
            i = 10
    "\n    Using drbdadm to show status of the DRBD devices,\n    available in the latest drbd9.\n    Support multiple nodes, multiple volumes.\n\n    :type name: str\n    :param name:\n        Resource name.\n\n    :return: drbd status of resource.\n    :rtype: list(dict(res))\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' drbd.status\n        salt '*' drbd.status name=<resource name>\n    "
    global ret
    global resource
    ret = []
    resource = {}
    cmd = ['drbdadm', 'status']
    cmd.append(name)
    for line in __salt__['cmd.run'](cmd).splitlines():
        _line_parser(line)
    if resource:
        ret.append(resource)
    return ret