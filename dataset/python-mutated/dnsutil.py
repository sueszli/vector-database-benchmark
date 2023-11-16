"""
Compendium of generic DNS utilities.

.. note::

    Some functions in the ``dnsutil`` execution module depend on ``dig``.
"""
import logging
import socket
import time
import salt.utils.files
import salt.utils.path
import salt.utils.stringutils
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Generic, should work on any platform (including Windows). Functionality\n    which requires dependencies outside of Python do not belong in this module.\n    '
    return True

def parse_hosts(hostsfile='/etc/hosts', hosts=None):
    if False:
        while True:
            i = 10
    "\n    Parse /etc/hosts file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dnsutil.parse_hosts\n    "
    if not hosts:
        try:
            with salt.utils.files.fopen(hostsfile, 'r') as fp_:
                hosts = salt.utils.stringutils.to_unicode(fp_.read())
        except Exception:
            return 'Error: hosts data was not found'
    hostsdict = {}
    for line in hosts.splitlines():
        if not line:
            continue
        if line.startswith('#'):
            continue
        comps = line.split()
        ip = comps[0]
        aliases = comps[1:]
        hostsdict.setdefault(ip, []).extend(aliases)
    return hostsdict

def hosts_append(hostsfile='/etc/hosts', ip_addr=None, entries=None):
    if False:
        print('Hello World!')
    "\n    Append a single line to the /etc/hosts file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dnsutil.hosts_append /etc/hosts 127.0.0.1 ad1.yuk.co,ad2.yuk.co\n    "
    host_list = entries.split(',')
    hosts = parse_hosts(hostsfile=hostsfile)
    if ip_addr in hosts:
        for host in host_list:
            if host in hosts[ip_addr]:
                host_list.remove(host)
    if not host_list:
        return 'No additional hosts were added to {}'.format(hostsfile)
    append_line = '\n{} {}'.format(ip_addr, ' '.join(host_list))
    with salt.utils.files.fopen(hostsfile, 'a') as fp_:
        fp_.write(salt.utils.stringutils.to_str(append_line))
    return 'The following line was added to {}:{}'.format(hostsfile, append_line)

def hosts_remove(hostsfile='/etc/hosts', entries=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove a host from the /etc/hosts file. If doing so will leave a line\n    containing only an IP address, then the line will be deleted. This function\n    will leave comments and blank lines intact.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' dnsutil.hosts_remove /etc/hosts ad1.yuk.co\n        salt '*' dnsutil.hosts_remove /etc/hosts ad2.yuk.co,ad1.yuk.co\n    "
    with salt.utils.files.fopen(hostsfile, 'r') as fp_:
        hosts = salt.utils.stringutils.to_unicode(fp_.read())
    host_list = entries.split(',')
    with salt.utils.files.fopen(hostsfile, 'w') as out_file:
        for line in hosts.splitlines():
            if not line or line.strip().startswith('#'):
                out_file.write(salt.utils.stringutils.to_str('{}\n'.format(line)))
                continue
            comps = line.split()
            for host in host_list:
                if host in comps[1:]:
                    comps.remove(host)
            if len(comps) > 1:
                out_file.write(salt.utils.stringutils.to_str(' '.join(comps)))
                out_file.write(salt.utils.stringutils.to_str('\n'))

def parse_zone(zonefile=None, zone=None):
    if False:
        while True:
            i = 10
    '\n    Parses a zone file. Can be passed raw zone data on the API level.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt ns1 dnsutil.parse_zone /var/lib/named/example.com.zone\n    '
    if zonefile:
        try:
            with salt.utils.files.fopen(zonefile, 'r') as fp_:
                zone = salt.utils.stringutils.to_unicode(fp_.read())
        except Exception:
            pass
    if not zone:
        return 'Error: Zone data was not found'
    zonedict = {}
    mode = 'single'
    for line in zone.splitlines():
        comps = line.split(';')
        line = comps[0].strip()
        if not line:
            continue
        comps = line.split()
        if line.startswith('$'):
            zonedict[comps[0].replace('$', '')] = comps[1]
            continue
        if '(' in line and ')' not in line:
            mode = 'multi'
            multi = ''
        if mode == 'multi':
            multi += ' {}'.format(line)
            if ')' in line:
                mode = 'single'
                line = multi.replace('(', '').replace(')', '')
            else:
                continue
        if 'ORIGIN' in zonedict:
            comps = line.replace('@', zonedict['ORIGIN']).split()
        else:
            comps = line.split()
        if 'SOA' in line:
            if comps[1] != 'IN':
                comps.pop(1)
            zonedict['ORIGIN'] = comps[0]
            zonedict['NETWORK'] = comps[1]
            zonedict['SOURCE'] = comps[3]
            zonedict['CONTACT'] = comps[4].replace('.', '@', 1)
            zonedict['SERIAL'] = comps[5]
            zonedict['REFRESH'] = _to_seconds(comps[6])
            zonedict['RETRY'] = _to_seconds(comps[7])
            zonedict['EXPIRE'] = _to_seconds(comps[8])
            zonedict['MINTTL'] = _to_seconds(comps[9])
            continue
        if comps[0] == 'IN':
            comps.insert(0, zonedict['ORIGIN'])
        if not comps[0].endswith('.') and 'NS' not in line:
            comps[0] = '{}.{}'.format(comps[0], zonedict['ORIGIN'])
        if comps[2] == 'NS':
            zonedict.setdefault('NS', []).append(comps[3])
        elif comps[2] == 'MX':
            if 'MX' not in zonedict:
                zonedict.setdefault('MX', []).append({'priority': comps[3], 'host': comps[4]})
        elif comps[3] in ('A', 'AAAA'):
            zonedict.setdefault(comps[3], {})[comps[0]] = {'TARGET': comps[4], 'TTL': comps[1]}
        else:
            zonedict.setdefault(comps[2], {})[comps[0]] = comps[3]
    return zonedict

def _to_seconds(timestr):
    if False:
        return 10
    '\n    Converts a time value to seconds.\n\n    As per RFC1035 (page 45), max time is 1 week, so anything longer (or\n    unreadable) will be set to one week (604800 seconds).\n    '
    timestr = timestr.upper()
    if 'H' in timestr:
        seconds = int(timestr.replace('H', '')) * 3600
    elif 'D' in timestr:
        seconds = int(timestr.replace('D', '')) * 86400
    elif 'W' in timestr:
        seconds = 604800
    else:
        try:
            seconds = int(timestr)
        except ValueError:
            seconds = 604800
    if seconds > 604800:
        seconds = 604800
    return seconds

def _has_dig():
    if False:
        i = 10
        return i + 15
    '\n    The dig-specific functions have been moved into their own module, but\n    because they are also DNS utilities, a compatibility layer exists. This\n    function helps add that layer.\n    '
    return salt.utils.path.which('dig') is not None

def check_ip(ip_addr):
    if False:
        return 10
    '\n    Check that string ip_addr is a valid IP\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt ns1 dnsutil.check_ip 127.0.0.1\n    '
    if _has_dig():
        return __salt__['dig.check_ip'](ip_addr)
    return 'This function requires dig, which is not currently available'

def A(host, nameserver=None):
    if False:
        i = 10
        return i + 15
    '\n    Return the A record(s) for ``host``.\n\n    Always returns a list.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt ns1 dnsutil.A www.google.com\n    '
    if _has_dig():
        return __salt__['dig.A'](host, nameserver)
    elif nameserver is None:
        try:
            addresses = [sock[4][0] for sock in socket.getaddrinfo(host, None, socket.AF_INET, 0, socket.SOCK_RAW)]
            return addresses
        except socket.gaierror:
            return 'Unable to resolve {}'.format(host)
    return 'This function requires dig, which is not currently available'

def AAAA(host, nameserver=None):
    if False:
        while True:
            i = 10
    '\n    Return the AAAA record(s) for ``host``.\n\n    Always returns a list.\n\n    .. versionadded:: 2014.7.5\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt ns1 dnsutil.AAAA www.google.com\n    '
    if _has_dig():
        return __salt__['dig.AAAA'](host, nameserver)
    elif nameserver is None:
        try:
            addresses = [sock[4][0] for sock in socket.getaddrinfo(host, None, socket.AF_INET6, 0, socket.SOCK_RAW)]
            return addresses
        except socket.gaierror:
            return 'Unable to resolve {}'.format(host)
    return 'This function requires dig, which is not currently available'

def NS(domain, resolve=True, nameserver=None):
    if False:
        return 10
    "\n    Return a list of IPs of the nameservers for ``domain``\n\n    If 'resolve' is False, don't resolve names.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt ns1 dnsutil.NS google.com\n\n    "
    if _has_dig():
        return __salt__['dig.NS'](domain, resolve, nameserver)
    return 'This function requires dig, which is not currently available'

def SPF(domain, record='SPF', nameserver=None):
    if False:
        i = 10
        return i + 15
    '\n    Return the allowed IPv4 ranges in the SPF record for ``domain``.\n\n    If record is ``SPF`` and the SPF record is empty, the TXT record will be\n    searched automatically. If you know the domain uses TXT and not SPF,\n    specifying that will save a lookup.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt ns1 dnsutil.SPF google.com\n    '
    if _has_dig():
        return __salt__['dig.SPF'](domain, record, nameserver)
    return 'This function requires dig, which is not currently available'

def MX(domain, resolve=False, nameserver=None):
    if False:
        while True:
            i = 10
    "\n    Return a list of lists for the MX of ``domain``.\n\n    If the 'resolve' argument is True, resolve IPs for the servers.\n\n    It's limited to one IP, because although in practice it's very rarely a\n    round robin, it is an acceptable configuration and pulling just one IP lets\n    the data be similar to the non-resolved version. If you think an MX has\n    multiple IPs, don't use the resolver here, resolve them in a separate step.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt ns1 dnsutil.MX google.com\n    "
    if _has_dig():
        return __salt__['dig.MX'](domain, resolve, nameserver)
    return 'This function requires dig, which is not currently available'

def serial(zone='', update=False):
    if False:
        return 10
    '\n    Return, store and update a dns serial for your zone files.\n\n    zone: a keyword for a specific zone\n\n    update: store an updated version of the serial in a grain\n\n    If ``update`` is False, the function will retrieve an existing serial or\n    return the current date if no serial is stored. Nothing will be stored\n\n    If ``update`` is True, the function will set the serial to the current date\n    if none exist or if the existing serial is for a previous date. If a serial\n    for greater than the current date is already stored, the function will\n    increment it.\n\n    This module stores the serial in a grain, you can explicitly set the\n    stored value as a grain named ``dnsserial_<zone_name>``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt ns1 dnsutil.serial example.com\n    '
    grains = {}
    key = 'dnsserial'
    if zone:
        key += '_{}'.format(zone)
    stored = __salt__['grains.get'](key=key)
    present = time.strftime('%Y%m%d01')
    if not update:
        return stored or present
    if stored and stored >= present:
        current = str(int(stored) + 1)
    else:
        current = present
    __salt__['grains.setval'](key=key, val=current)
    return current