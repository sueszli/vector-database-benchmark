"""
Support for Tomcat

This module uses the manager webapp to manage Apache tomcat webapps.
If the manager webapp is not configured some of the functions won't work.

:configuration:
    - Java bin path should be in default path
    - If ipv6 is enabled make sure you permit manager access to ipv6 interface
      "0:0:0:0:0:0:0:1"
    - If you are using tomcat.tar.gz it has to be installed or symlinked under
      ``/opt``, preferably using name tomcat
    - "tomcat.signal start/stop" works but it does not use the startup scripts

The following grains/pillar should be set:

.. code-block:: yaml

    tomcat-manager:
      user: <username>
      passwd: <password>

or the old format:

.. code-block:: yaml

    tomcat-manager.user: <username>
    tomcat-manager.passwd: <password>

Also configure a user in the conf/tomcat-users.xml file:

.. code-block:: xml

    <?xml version='1.0' encoding='utf-8'?>
    <tomcat-users>
        <role rolename="manager-script"/>
        <user username="tomcat" password="tomcat" roles="manager-script"/>
    </tomcat-users>

.. note::

   - More information about tomcat manager:
     http://tomcat.apache.org/tomcat-7.0-doc/manager-howto.html
   - if you use only this module for deployments you've might want to strict
     access to the manager only from localhost for more info:
     http://tomcat.apache.org/tomcat-7.0-doc/manager-howto.html#Configuring_Manager_Application_Access
   - Tested on:

     JVM Vendor:
         Sun Microsystems Inc.
     JVM Version:
         1.6.0_43-b01
     OS Architecture:
         amd64
     OS Name:
         Linux
     OS Version:
         2.6.32-358.el6.x86_64
     Tomcat Version:
         Apache Tomcat/7.0.37
"""
import glob
import hashlib
import logging
import os
import re
import tempfile
import urllib.parse
import urllib.request
import salt.utils.data
import salt.utils.stringutils
log = logging.getLogger(__name__)
__func_alias__ = {'reload_': 'reload'}
__valid_configs = {'user': ['tomcat-manager.user', 'tomcat-manager:user'], 'passwd': ['tomcat-manager.passwd', 'tomcat-manager:passwd']}

def __virtual__():
    if False:
        return 10
    '\n    Only load tomcat if it is installed or if grains/pillar config exists\n    '
    if __catalina_home() or _auth('dummy'):
        return 'tomcat'
    return (False, 'Tomcat execution module not loaded: neither Tomcat installed locally nor tomcat-manager credentials set in grains/pillar/config.')

def __catalina_home():
    if False:
        return 10
    '\n    Tomcat paths differ depending on packaging\n    '
    locations = ['/usr/share/tomcat*', '/opt/tomcat']
    for location in locations:
        folders = glob.glob(location)
        if folders:
            for catalina_home in folders:
                if os.path.isdir(catalina_home + '/bin'):
                    return catalina_home
    return False

def _get_credentials():
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the username and password from opts, grains, or pillar\n    '
    ret = {'user': False, 'passwd': False}
    for item in ret:
        for struct in [__opts__, __grains__, __pillar__]:
            for config_key in __valid_configs[item]:
                value = salt.utils.data.traverse_dict_and_list(struct, config_key, None)
                if value:
                    ret[item] = value
                    break
    return (ret['user'], ret['passwd'])

def _auth(uri):
    if False:
        for i in range(10):
            print('nop')
    '\n    returns a authentication handler.\n    Get user & password from grains, if are not set default to\n    modules.config.option\n\n    If user & pass are missing return False\n    '
    (user, password) = _get_credentials()
    if user is False or password is False:
        return False
    basic = urllib.request.HTTPBasicAuthHandler()
    basic.add_password(realm='Tomcat Manager Application', uri=uri, user=user, passwd=password)
    digest = urllib.request.HTTPDigestAuthHandler()
    digest.add_password(realm='Tomcat Manager Application', uri=uri, user=user, passwd=password)
    return urllib.request.build_opener(basic, digest)

def extract_war_version(war):
    if False:
        while True:
            i = 10
    '\n    Extract the version from the war file name. There does not seem to be a\n    standard for encoding the version into the `war file name`_\n\n    .. _`war file name`: https://tomcat.apache.org/tomcat-6.0-doc/deployer-howto.html\n\n    Examples:\n\n    .. code-block:: bash\n\n        /path/salt-2015.8.6.war -> 2015.8.6\n        /path/V6R2013xD5.war -> None\n    '
    basename = os.path.basename(war)
    war_package = os.path.splitext(basename)[0]
    version = re.findall('-([\\d.-]+)$', war_package)
    return version[0] if version and len(version) == 1 else None

def _wget(cmd, opts=None, url='http://localhost:8080/manager', timeout=180):
    if False:
        for i in range(10):
            print('nop')
    '\n    A private function used to issue the command to tomcat via the manager\n    webapp\n\n    cmd\n        the command to execute\n\n    url\n        The URL of the server manager webapp (example:\n        http://localhost:8080/manager)\n\n    opts\n        a dict of arguments\n\n    timeout\n        timeout for HTTP request\n\n    Return value is a dict in the from of::\n\n        {\n            res: [True|False]\n            msg: list of lines we got back from the manager\n        }\n    '
    ret = {'res': True, 'msg': []}
    auth = _auth(url)
    if auth is False:
        ret['res'] = False
        ret['msg'] = 'missing username and password settings (grain/pillar)'
        return ret
    if url[-1] != '/':
        url += '/'
    url6 = url
    url += 'text/{}'.format(cmd)
    url6 += '{}'.format(cmd)
    if opts:
        url += '?{}'.format(urllib.parse.urlencode(opts))
        url6 += '?{}'.format(urllib.parse.urlencode(opts))
    urllib.request.install_opener(auth)
    try:
        ret['msg'] = urllib.request.urlopen(url, timeout=timeout).read().splitlines()
    except Exception:
        try:
            ret['msg'] = urllib.request.urlopen(url6, timeout=timeout).read().splitlines()
        except Exception:
            ret['msg'] = 'Failed to create HTTP request'
    for (key, value) in enumerate(ret['msg']):
        try:
            ret['msg'][key] = salt.utils.stringutils.to_unicode(value, 'utf-8')
        except (UnicodeDecodeError, AttributeError):
            pass
    if not ret['msg'][0].startswith('OK'):
        ret['res'] = False
    return ret

def _simple_cmd(cmd, app, url='http://localhost:8080/manager', timeout=180):
    if False:
        for i in range(10):
            print('nop')
    '\n    Simple command wrapper to commands that need only a path option\n    '
    try:
        opts = {'path': app, 'version': ls(url)[app]['version']}
        return '\n'.join(_wget(cmd, opts, url, timeout=timeout)['msg'])
    except Exception:
        return 'FAIL - No context exists for path {}'.format(app)

def leaks(url='http://localhost:8080/manager', timeout=180):
    if False:
        return 10
    "\n    Find memory leaks in tomcat\n\n    url : http://localhost:8080/manager\n        the URL of the server manager webapp\n    timeout : 180\n        timeout for HTTP request\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' tomcat.leaks\n    "
    return _wget('findleaks', {'statusLine': 'true'}, url, timeout=timeout)['msg']

def status(url='http://localhost:8080/manager', timeout=180):
    if False:
        for i in range(10):
            print('nop')
    "\n    Used to test if the tomcat manager is up\n\n    url : http://localhost:8080/manager\n        the URL of the server manager webapp\n    timeout : 180\n        timeout for HTTP request\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' tomcat.status\n        salt '*' tomcat.status http://localhost:8080/manager\n    "
    return _wget('list', {}, url, timeout=timeout)['res']

def ls(url='http://localhost:8080/manager', timeout=180):
    if False:
        return 10
    "\n    list all the deployed webapps\n\n    url : http://localhost:8080/manager\n        the URL of the server manager webapp\n    timeout : 180\n        timeout for HTTP request\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' tomcat.ls\n        salt '*' tomcat.ls http://localhost:8080/manager\n    "
    ret = {}
    data = _wget('list', '', url, timeout=timeout)
    if data['res'] is False:
        return {}
    data['msg'].pop(0)
    for line in data['msg']:
        tmp = line.split(':')
        ret[tmp[0]] = {'mode': tmp[1], 'sessions': tmp[2], 'fullname': tmp[3], 'version': ''}
        sliced = tmp[3].split('##')
        if len(sliced) > 1:
            ret[tmp[0]]['version'] = sliced[1]
    return ret

def stop(app, url='http://localhost:8080/manager', timeout=180):
    if False:
        for i in range(10):
            print('nop')
    "\n    Stop the webapp\n\n    app\n        the webapp context path\n    url : http://localhost:8080/manager\n        the URL of the server manager webapp\n    timeout : 180\n        timeout for HTTP request\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' tomcat.stop /jenkins\n        salt '*' tomcat.stop /jenkins http://localhost:8080/manager\n    "
    return _simple_cmd('stop', app, url, timeout=timeout)

def start(app, url='http://localhost:8080/manager', timeout=180):
    if False:
        for i in range(10):
            print('nop')
    "\n    Start the webapp\n\n    app\n        the webapp context path\n    url : http://localhost:8080/manager\n        the URL of the server manager webapp\n    timeout\n        timeout for HTTP request\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' tomcat.start /jenkins\n        salt '*' tomcat.start /jenkins http://localhost:8080/manager\n    "
    return _simple_cmd('start', app, url, timeout=timeout)

def reload_(app, url='http://localhost:8080/manager', timeout=180):
    if False:
        while True:
            i = 10
    "\n    Reload the webapp\n\n    app\n        the webapp context path\n    url : http://localhost:8080/manager\n        the URL of the server manager webapp\n    timeout : 180\n        timeout for HTTP request\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' tomcat.reload /jenkins\n        salt '*' tomcat.reload /jenkins http://localhost:8080/manager\n    "
    return _simple_cmd('reload', app, url, timeout=timeout)

def sessions(app, url='http://localhost:8080/manager', timeout=180):
    if False:
        print('Hello World!')
    "\n    return the status of the webapp sessions\n\n    app\n        the webapp context path\n    url : http://localhost:8080/manager\n        the URL of the server manager webapp\n    timeout : 180\n        timeout for HTTP request\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' tomcat.sessions /jenkins\n        salt '*' tomcat.sessions /jenkins http://localhost:8080/manager\n    "
    return _simple_cmd('sessions', app, url, timeout=timeout)

def status_webapp(app, url='http://localhost:8080/manager', timeout=180):
    if False:
        for i in range(10):
            print('nop')
    "\n    return the status of the webapp (stopped | running | missing)\n\n    app\n        the webapp context path\n    url : http://localhost:8080/manager\n        the URL of the server manager webapp\n    timeout : 180\n        timeout for HTTP request\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' tomcat.status_webapp /jenkins\n        salt '*' tomcat.status_webapp /jenkins http://localhost:8080/manager\n    "
    webapps = ls(url, timeout=timeout)
    for i in webapps:
        if i == app:
            return webapps[i]['mode']
    return 'missing'

def serverinfo(url='http://localhost:8080/manager', timeout=180):
    if False:
        print('Hello World!')
    "\n    return details about the server\n\n    url : http://localhost:8080/manager\n        the URL of the server manager webapp\n    timeout : 180\n        timeout for HTTP request\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' tomcat.serverinfo\n        salt '*' tomcat.serverinfo http://localhost:8080/manager\n    "
    data = _wget('serverinfo', {}, url, timeout=timeout)
    if data['res'] is False:
        return {'error': data['msg']}
    ret = {}
    data['msg'].pop(0)
    for line in data['msg']:
        tmp = line.split(':')
        ret[tmp[0].strip()] = tmp[1].strip()
    return ret

def undeploy(app, url='http://localhost:8080/manager', timeout=180):
    if False:
        while True:
            i = 10
    "\n    Undeploy a webapp\n\n    app\n        the webapp context path\n    url : http://localhost:8080/manager\n        the URL of the server manager webapp\n    timeout : 180\n        timeout for HTTP request\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' tomcat.undeploy /jenkins\n        salt '*' tomcat.undeploy /jenkins http://localhost:8080/manager\n    "
    return _simple_cmd('undeploy', app, url, timeout=timeout)

def deploy_war(war, context, force='no', url='http://localhost:8080/manager', saltenv='base', timeout=180, temp_war_location=None, version=True):
    if False:
        print('Hello World!')
    "\n    Deploy a WAR file\n\n    war\n        absolute path to WAR file (should be accessible by the user running\n        tomcat) or a path supported by the salt.modules.cp.get_file function\n    context\n        the context path to deploy\n    force : False\n        set True to deploy the webapp even one is deployed in the context\n    url : http://localhost:8080/manager\n        the URL of the server manager webapp\n    saltenv : base\n        the environment for WAR file in used by salt.modules.cp.get_url\n        function\n    timeout : 180\n        timeout for HTTP request\n    temp_war_location : None\n        use another location to temporarily copy to war file\n        by default the system's temp directory is used\n    version : ''\n        Specify the war version.  If this argument is provided, it overrides\n        the version encoded in the war file name, if one is present.\n\n        Examples:\n\n        .. code-block:: bash\n\n            salt '*' tomcat.deploy_war salt://salt-2015.8.6.war version=2015.08.r6\n\n        .. versionadded:: 2015.8.6\n\n    CLI Examples:\n\n    cp module\n\n    .. code-block:: bash\n\n        salt '*' tomcat.deploy_war salt://application.war /api\n        salt '*' tomcat.deploy_war salt://application.war /api no\n        salt '*' tomcat.deploy_war salt://application.war /api yes http://localhost:8080/manager\n\n    minion local file system\n\n    .. code-block:: bash\n\n        salt '*' tomcat.deploy_war /tmp/application.war /api\n        salt '*' tomcat.deploy_war /tmp/application.war /api no\n        salt '*' tomcat.deploy_war /tmp/application.war /api yes http://localhost:8080/manager\n    "
    tfile = 'salt.{}'.format(os.path.basename(war))
    if temp_war_location is not None:
        if not os.path.isdir(temp_war_location):
            return 'Error - "{}" is not a directory'.format(temp_war_location)
        tfile = os.path.join(temp_war_location, tfile)
    else:
        tfile = os.path.join(tempfile.gettempdir(), tfile)
    cache = False
    if not os.path.isfile(war):
        cache = True
        cached = __salt__['cp.get_url'](war, tfile, saltenv)
        if not cached:
            return 'FAIL - could not cache the WAR file'
        try:
            __salt__['file.set_mode'](cached, '0644')
        except KeyError:
            pass
    else:
        tfile = war
    opts = {'war': 'file:{}'.format(tfile), 'path': context}
    if version:
        version = extract_war_version(war) if version is True else version
        if isinstance(version, str):
            opts['version'] = version
    if force == 'yes':
        opts['update'] = 'true'
    deployed = _wget('deploy', opts, url, timeout=timeout)
    res = '\n'.join(deployed['msg'])
    if cache:
        __salt__['file.remove'](tfile)
    return res

def passwd(passwd, user='', alg='sha1', realm=None):
    if False:
        return 10
    "\n    This function replaces the $CATALINA_HOME/bin/digest.sh script\n    convert a clear-text password to the $CATALINA_BASE/conf/tomcat-users.xml\n    format\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' tomcat.passwd secret\n        salt '*' tomcat.passwd secret tomcat sha1\n        salt '*' tomcat.passwd secret tomcat sha1 'Protected Realm'\n    "
    digest = hasattr(hashlib, alg) and getattr(hashlib, alg) or None
    if digest:
        if realm:
            digest.update('{}:{}:{}'.format(user, realm, passwd))
        else:
            digest.update(passwd)
    return digest and digest.hexdigest() or False

def version():
    if False:
        print('Hello World!')
    "\n    Return server version from catalina.sh version\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' tomcat.version\n    "
    cmd = __catalina_home() + '/bin/catalina.sh version'
    out = __salt__['cmd.run'](cmd).splitlines()
    for line in out:
        if not line:
            continue
        if 'Server version' in line:
            comps = line.split(': ')
            return comps[1]

def fullversion():
    if False:
        return 10
    "\n    Return all server information from catalina.sh version\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' tomcat.fullversion\n    "
    cmd = __catalina_home() + '/bin/catalina.sh version'
    ret = {}
    out = __salt__['cmd.run'](cmd).splitlines()
    for line in out:
        if not line:
            continue
        if ': ' in line:
            comps = line.split(': ')
            ret[comps[0]] = comps[1].lstrip()
    return ret

def signal(signal=None):
    if False:
        while True:
            i = 10
    "\n    Signals catalina to start, stop, securestart, forcestop.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' tomcat.signal start\n    "
    valid_signals = {'forcestop': 'stop -force', 'securestart': 'start -security', 'start': 'start', 'stop': 'stop'}
    if signal not in valid_signals:
        return
    cmd = '{}/bin/catalina.sh {}'.format(__catalina_home(), valid_signals[signal])
    __salt__['cmd.run'](cmd)
if __name__ == '__main__':
    __opts__ = {}
    __grains__ = {}
    __pillar__ = {'tomcat-manager.user': 'foobar', 'tomcat-manager.passwd': 'barfoo1!'}
    old_format_creds = _get_credentials()
    __pillar__ = {'tomcat-manager': {'user': 'foobar', 'passwd': 'barfoo1!'}}
    new_format_creds = _get_credentials()
    if old_format_creds == new_format_creds:
        log.info('Config backwards compatible')
    else:
        log.ifno('Config not backwards compatible')