"""
This files implement buildbotNetUsageData options
It uses urllib instead of requests in order to avoid requiring another dependency for statistics
feature.
urllib supports http_proxy already. urllib is blocking and thus everything is done from a thread.
"""
import hashlib
import inspect
import json
import os
import platform
import socket
from urllib import error as urllib_error
from urllib import request as urllib_request
from twisted.internet import threads
from twisted.python import log
from buildbot.process.buildstep import _BuildStepFactory
from buildbot.util import unicode2bytes
from buildbot.www.config import get_environment_versions
PHONE_HOME_URL = 'https://events.buildbot.net/events/phone_home'

def linux_distribution():
    if False:
        return 10
    os_release = '/etc/os-release'
    meta_data = {}
    if os.path.exists(os_release):
        with open('/etc/os-release', encoding='utf-8') as f:
            for line in f:
                try:
                    (k, v) = line.strip().split('=')
                    meta_data[k] = v.strip('""')
                except Exception:
                    pass
    linux_id = meta_data.get('ID', 'unknown_linux')
    linux_version = 'unknown_version'
    for version_key in ['VERSION_ID', 'VERSION_CODENAME']:
        linux_version = meta_data.get(version_key, linux_version)
    return (linux_id, linux_version)

def get_distro():
    if False:
        while True:
            i = 10
    system = platform.system()
    if system == 'Linux':
        dist = linux_distribution()
        return f'{dist[0]}:{dist[1]}'
    elif system == 'Windows':
        dist = platform.win32_ver()
        return f'{dist[0]}:{dist[1]}'
    elif system == 'Java':
        dist = platform.java_ver()
        return f'{dist[0]}:{dist[1]}'
    elif system == 'Darwin':
        dist = platform.mac_ver()
        return f'{dist[0]}'
    return ':'.join(platform.uname()[0:1])

def getName(obj):
    if False:
        for i in range(10):
            print('nop')
    'This method finds the first parent class which is within the buildbot namespace\n    it prepends the name with as many ">" as the class is subclassed\n    '

    def sanitize(name):
        if False:
            for i in range(10):
                print('nop')
        return name.replace('.', '/')
    if isinstance(obj, _BuildStepFactory):
        klass = obj.factory
    else:
        klass = type(obj)
    name = ''
    klasses = (klass,) + inspect.getmro(klass)
    for klass in klasses:
        if hasattr(klass, '__module__') and klass.__module__.startswith('buildbot.'):
            return sanitize(name + klass.__module__ + '.' + klass.__name__)
        else:
            name += '>'
    return sanitize(type(obj).__name__)

def countPlugins(plugins_uses, lst):
    if False:
        return 10
    if isinstance(lst, dict):
        lst = lst.values()
    for i in lst:
        name = getName(i)
        plugins_uses.setdefault(name, 0)
        plugins_uses[name] += 1

def basicData(master):
    if False:
        for i in range(10):
            print('nop')
    plugins_uses = {}
    countPlugins(plugins_uses, master.config.workers)
    countPlugins(plugins_uses, master.config.builders)
    countPlugins(plugins_uses, master.config.schedulers)
    countPlugins(plugins_uses, master.config.services)
    countPlugins(plugins_uses, master.config.change_sources)
    for b in master.config.builders:
        countPlugins(plugins_uses, b.factory.steps)
    hashInput = master.name + socket.getfqdn()
    hashInput = unicode2bytes(hashInput)
    installid = hashlib.sha1(hashInput).hexdigest()
    return {'installid': installid, 'versions': dict(get_environment_versions()), 'platform': {'platform': platform.platform(), 'system': platform.system(), 'machine': platform.machine(), 'processor': platform.processor(), 'python_implementation': platform.python_implementation(), 'version': ' '.join(platform.version().split(' ')[:4]), 'distro': get_distro()}, 'plugins': plugins_uses, 'db': master.config.db['db_url'].split('://')[0], 'mq': master.config.mq['type'], 'www_plugins': list(master.config.www['plugins'].keys())}

def fullData(master):
    if False:
        for i in range(10):
            print('nop')
    '\n        Send the actual configuration of the builders, how the steps are agenced.\n        Note that full data will never send actual detail of what command is run, name of servers,\n        etc.\n    '
    builders = []
    for b in master.config.builders:
        steps = []
        for step in b.factory.steps:
            steps.append(getName(step))
        builders.append(steps)
    return {'builders': builders}

def computeUsageData(master):
    if False:
        for i in range(10):
            print('nop')
    if master.config.buildbotNetUsageData is None:
        return None
    data = basicData(master)
    if master.config.buildbotNetUsageData != 'basic':
        data.update(fullData(master))
    if callable(master.config.buildbotNetUsageData):
        data = master.config.buildbotNetUsageData(data)
    return data

def _sendWithUrlib(url, data):
    if False:
        for i in range(10):
            print('nop')
    data = json.dumps(data).encode()
    clen = len(data)
    req = urllib_request.Request(url, data, {'Content-Type': 'application/json', 'Content-Length': clen})
    try:
        f = urllib_request.urlopen(req)
    except urllib_error.URLError:
        return None
    res = f.read()
    f.close()
    return res

def _sendWithRequests(url, data):
    if False:
        while True:
            i = 10
    try:
        import requests
    except ImportError:
        return None
    r = requests.post(url, json=data, timeout=30)
    return r.text

def _sendBuildbotNetUsageData(data):
    if False:
        for i in range(10):
            print('nop')
    log.msg(f'buildbotNetUsageData: sending {data}')
    res = _sendWithRequests(PHONE_HOME_URL, data)
    if res is None:
        res = _sendWithUrlib(PHONE_HOME_URL, data)
    if res is None:
        log.msg("buildbotNetUsageData: Could not send using https, please `pip install 'requests[security]'` for proper SSL implementation`")
        data['buggySSL'] = True
        res = _sendWithUrlib(PHONE_HOME_URL.replace('https://', 'http://'), data)
    log.msg('buildbotNetUsageData: buildbot.net said:', res)

def sendBuildbotNetUsageData(master):
    if False:
        i = 10
        return i + 15
    if master.config.buildbotNetUsageData is None:
        return
    data = computeUsageData(master)
    if data is None:
        return
    threads.deferToThread(_sendBuildbotNetUsageData, data)