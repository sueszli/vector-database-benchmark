"""
Support for Apache

.. note::
    The functions in here are generic functions designed to work with
    all implementations of Apache. Debian-specific functions have been moved into
    deb_apache.py, but will still load under the ``apache`` namespace when a
    Debian-based system is detected.
"""
import io
import logging
import re
import urllib.error
import urllib.request
import salt.utils.data
import salt.utils.files
import salt.utils.path
import salt.utils.stringutils
from salt.exceptions import SaltException
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load the module if apache is installed\n    '
    cmd = _detect_os()
    if salt.utils.path.which(cmd):
        return 'apache'
    return (False, 'The apache execution module cannot be loaded: apache is not installed.')

def _detect_os():
    if False:
        while True:
            i = 10
    '\n    Apache commands and paths differ depending on packaging\n    '
    os_family = __grains__['os_family']
    if os_family == 'RedHat':
        return 'apachectl'
    elif os_family == 'Debian' or os_family == 'Suse':
        return 'apache2ctl'
    else:
        return 'apachectl'

def version():
    if False:
        print('Hello World!')
    "\n    Return server version (``apachectl -v``)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apache.version\n    "
    cmd = '{} -v'.format(_detect_os())
    out = __salt__['cmd.run'](cmd).splitlines()
    ret = out[0].split(': ')
    return ret[1]

def fullversion():
    if False:
        print('Hello World!')
    "\n    Return server version (``apachectl -V``)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apache.fullversion\n    "
    cmd = '{} -V'.format(_detect_os())
    ret = {}
    ret['compiled_with'] = []
    out = __salt__['cmd.run'](cmd).splitlines()
    define_re = re.compile('^\\s+-D\\s+')
    for line in out:
        if ': ' in line:
            comps = line.split(': ')
            if not comps:
                continue
            ret[comps[0].strip().lower().replace(' ', '_')] = comps[1].strip()
        elif ' -D' in line:
            cwith = define_re.sub('', line)
            ret['compiled_with'].append(cwith)
    return ret

def modules():
    if False:
        i = 10
        return i + 15
    "\n    Return list of static and shared modules (``apachectl -M``)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apache.modules\n    "
    cmd = '{} -M'.format(_detect_os())
    ret = {}
    ret['static'] = []
    ret['shared'] = []
    out = __salt__['cmd.run'](cmd).splitlines()
    for line in out:
        comps = line.split()
        if not comps:
            continue
        if '(static)' in line:
            ret['static'].append(comps[0])
        if '(shared)' in line:
            ret['shared'].append(comps[0])
    return ret

def servermods():
    if False:
        print('Hello World!')
    "\n    Return list of modules compiled into the server (``apachectl -l``)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apache.servermods\n    "
    cmd = '{} -l'.format(_detect_os())
    ret = []
    out = __salt__['cmd.run'](cmd).splitlines()
    for line in out:
        if not line:
            continue
        if '.c' in line:
            ret.append(line.strip())
    return ret

def directives():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return list of directives together with expected arguments\n    and places where the directive is valid (``apachectl -L``)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apache.directives\n    "
    cmd = '{} -L'.format(_detect_os())
    ret = {}
    out = __salt__['cmd.run'](cmd)
    out = out.replace('\n\t', '\t')
    for line in out.splitlines():
        if not line:
            continue
        comps = line.split('\t')
        desc = '\n'.join(comps[1:])
        ret[comps[0]] = desc
    return ret

def vhosts():
    if False:
        print('Hello World!')
    "\n    Show the settings as parsed from the config file (currently\n    only shows the virtualhost settings) (``apachectl -S``).\n    Because each additional virtual host adds to the execution\n    time, this command may require a long timeout be specified\n    by using ``-t 10``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt -t 10 '*' apache.vhosts\n    "
    cmd = '{} -S'.format(_detect_os())
    ret = {}
    namevhost = ''
    out = __salt__['cmd.run'](cmd)
    for line in out.splitlines():
        if not line:
            continue
        comps = line.split()
        if 'is a NameVirtualHost' in line:
            namevhost = comps[0]
            ret[namevhost] = {}
        else:
            if comps[0] == 'default':
                ret[namevhost]['default'] = {}
                ret[namevhost]['default']['vhost'] = comps[2]
                ret[namevhost]['default']['conf'] = re.sub('\\(|\\)', '', comps[3])
            if comps[0] == 'port':
                ret[namevhost][comps[3]] = {}
                ret[namevhost][comps[3]]['vhost'] = comps[3]
                ret[namevhost][comps[3]]['conf'] = re.sub('\\(|\\)', '', comps[4])
                ret[namevhost][comps[3]]['port'] = comps[1]
    return ret

def signal(signal=None):
    if False:
        i = 10
        return i + 15
    "\n    Signals httpd to start, restart, or stop.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apache.signal restart\n    "
    no_extra_args = ('configtest', 'status', 'fullstatus')
    valid_signals = ('start', 'stop', 'restart', 'graceful', 'graceful-stop')
    if signal not in valid_signals and signal not in no_extra_args:
        return
    if signal in valid_signals:
        arguments = ' -k {}'.format(signal)
    else:
        arguments = ' {}'.format(signal)
    cmd = _detect_os() + arguments
    out = __salt__['cmd.run_all'](cmd)
    if out['retcode'] and out['stderr']:
        ret = out['stderr'].strip()
    elif out['stderr']:
        ret = out['stderr'].strip()
    elif out['stdout']:
        ret = out['stdout'].strip()
    else:
        ret = 'Command: "{}" completed successfully!'.format(cmd)
    return ret

def useradd(pwfile, user, password, opts=''):
    if False:
        while True:
            i = 10
    "\n    Add HTTP user using the ``htpasswd`` command. If the ``htpasswd`` file does not\n    exist, it will be created. Valid options that can be passed are:\n\n    .. code-block:: text\n\n        n  Don't update file; display results on stdout.\n        m  Force MD5 hashing of the password (default).\n        d  Force CRYPT(3) hashing of the password.\n        p  Do not hash the password (plaintext).\n        s  Force SHA1 hashing of the password.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' apache.useradd /etc/httpd/htpasswd larry badpassword\n        salt '*' apache.useradd /etc/httpd/htpasswd larry badpass opts=ns\n    "
    return __salt__['webutil.useradd'](pwfile, user, password, opts)

def userdel(pwfile, user):
    if False:
        i = 10
        return i + 15
    "\n    Delete HTTP user from the specified ``htpasswd`` file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apache.userdel /etc/httpd/htpasswd larry\n    "
    return __salt__['webutil.userdel'](pwfile, user)

def server_status(profile='default'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get Information from the Apache server-status handler\n\n    .. note::\n\n        The server-status handler is disabled by default.\n        In order for this function to work it needs to be enabled.\n        See http://httpd.apache.org/docs/2.2/mod/mod_status.html\n\n    The following configuration needs to exists in pillar/grains.\n    Each entry nested in ``apache.server-status`` is a profile of a vhost/server.\n    This would give support for multiple apache servers/vhosts.\n\n    .. code-block:: yaml\n\n        apache.server-status:\n          default:\n            url: http://localhost/server-status\n            user: someuser\n            pass: password\n            realm: 'authentication realm for digest passwords'\n            timeout: 5\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' apache.server_status\n        salt '*' apache.server_status other-profile\n    "
    ret = {'Scoreboard': {'_': 0, 'S': 0, 'R': 0, 'W': 0, 'K': 0, 'D': 0, 'C': 0, 'L': 0, 'G': 0, 'I': 0, '.': 0}}
    url = __salt__['config.get']('apache.server-status:{}:url'.format(profile), 'http://localhost/server-status')
    user = __salt__['config.get']('apache.server-status:{}:user'.format(profile), '')
    passwd = __salt__['config.get']('apache.server-status:{}:pass'.format(profile), '')
    realm = __salt__['config.get']('apache.server-status:{}:realm'.format(profile), '')
    timeout = __salt__['config.get']('apache.server-status:{}:timeout'.format(profile), 5)
    if user and passwd:
        basic = urllib.request.HTTPBasicAuthHandler()
        basic.add_password(realm=realm, uri=url, user=user, passwd=passwd)
        digest = urllib.request.HTTPDigestAuthHandler()
        digest.add_password(realm=realm, uri=url, user=user, passwd=passwd)
        urllib.request.install_opener(urllib.request.build_opener(basic, digest))
    url += '?auto'
    try:
        response = urllib.request.urlopen(url, timeout=timeout).read().splitlines()
    except urllib.error.URLError:
        return 'error'
    for line in response:
        splt = line.split(':', 1)
        splt[0] = splt[0].strip()
        splt[1] = splt[1].strip()
        if splt[0] == 'Scoreboard':
            for c in splt[1]:
                ret['Scoreboard'][c] += 1
        elif splt[1].isdigit():
            ret[splt[0]] = int(splt[1])
        else:
            ret[splt[0]] = float(splt[1])
    return ret

def _parse_config(conf, slot=None):
    if False:
        return 10
    '\n    Recursively goes through config structure and builds final Apache configuration\n\n    :param conf: defined config structure\n    :param slot: name of section container if needed\n    '
    ret = io.StringIO()
    if isinstance(conf, str):
        if slot:
            print('{} {}'.format(slot, conf), file=ret, end='')
        else:
            print('{}'.format(conf), file=ret, end='')
    elif isinstance(conf, list):
        is_section = False
        for item in conf:
            if 'this' in item:
                is_section = True
                slot_this = str(item['this'])
        if is_section:
            print('<{} {}>'.format(slot, slot_this), file=ret)
            for item in conf:
                for (key, val) in item.items():
                    if key != 'this':
                        print(_parse_config(val, str(key)), file=ret)
            print('</{}>'.format(slot), file=ret)
        else:
            for value in conf:
                print(_parse_config(value, str(slot)), file=ret)
    elif isinstance(conf, dict):
        try:
            print('<{} {}>'.format(slot, conf['this']), file=ret)
        except KeyError:
            raise SaltException('Apache section container "<{}>" expects attribute. Specify it using key "this".'.format(slot))
        for (key, value) in conf.items():
            if key != 'this':
                if isinstance(value, str):
                    print('{} {}'.format(key, value), file=ret)
                elif isinstance(value, list):
                    print(_parse_config(value, key), file=ret)
                elif isinstance(value, dict):
                    print(_parse_config(value, key), file=ret)
        print('</{}>'.format(slot), file=ret)
    ret.seek(0)
    return ret.read()

def config(name, config, edit=True):
    if False:
        while True:
            i = 10
    '\n    Create VirtualHost configuration files\n\n    name\n        File for the virtual host\n    config\n        VirtualHost configurations\n\n    .. note::\n\n        This function is not meant to be used from the command line.\n        Config is meant to be an ordered dict of all of the apache configs.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' apache.config /etc/httpd/conf.d/ports.conf config="[{\'Listen\': \'22\'}]"\n    '
    configs = []
    for entry in config:
        key = next(iter(entry.keys()))
        configs.append(_parse_config(entry[key], key))
    configstext = '\n'.join(salt.utils.data.decode(configs))
    if edit:
        with salt.utils.files.fopen(name, 'w') as configfile:
            configfile.write('# This file is managed by Salt.\n')
            configfile.write(salt.utils.stringutils.to_str(configstext))
    return configstext