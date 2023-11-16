import re
import subprocess
import requests
import urllib
import os
from persepolis.scripts.useful_tools import osAndDesktopEnvironment
from persepolis.scripts import logger
from persepolis.constants import OS

def getProxy():
    if False:
        return 10
    socks_proxy = False
    (os_type, desktop) = osAndDesktopEnvironment()
    tmp = re.search('.*:(.*)', desktop)
    if tmp is not None:
        desktop = tmp.group(1)
    platform = 'platform : ' + os_type
    logger.sendToLog(platform, 'INFO')
    proxy = {}
    if os_type in OS.UNIX_LIKE:
        if desktop == None:
            desktop_env_type = 'Desktop Environment not detected!'
        else:
            desktop_env_type = 'Desktop environment: ' + str(desktop)
        logger.sendToLog(desktop_env_type, 'INFO')
    if desktop == 'KDE':
        proxysource = {}
        home_address = os.path.expanduser('~')
        try:
            plasma_proxy_config_file_path = os.path.join(home_address, '.config', 'kioslaverc')
        except:
            logger.sendToLog('no proxy file detected', 'INFO')
        if os.path.isfile(plasma_proxy_config_file_path):
            try:
                with open(plasma_proxy_config_file_path) as proxyfile:
                    for line in proxyfile:
                        (name, var) = line.partition('=')[::2]
                        proxysource[name.strip()] = str(var)
            except:
                logger.sendToLog('no proxy file detected', 'INFO')
            if proxysource['ProxyType'].split('\n')[0] == '1':
                try:
                    proxy['ftp_proxy_port'] = proxysource['ftpProxy'].split(' ')[1].replace('/', '').replace('\n', '')
                    proxy['ftp_proxy_ip'] = proxysource['ftpProxy'].split(' ')[0].split('//')[1]
                except:
                    logger.sendToLog('no manual ftp proxy detected', 'INFO')
                try:
                    proxy['http_proxy_port'] = proxysource['httpProxy'].split(' ')[1].replace('/', '').replace('\n', '')
                    proxy['http_proxy_ip'] = proxysource['httpProxy'].split(' ')[0].split('//')[1]
                except:
                    logger.sendToLog('no manual http proxy detected', 'INFO')
                try:
                    proxy['https_proxy_port'] = proxysource['httpsProxy'].split(' ')[1].replace('/', '').replace('\n', '')
                    proxy['https_proxy_ip'] = proxysource['httpsProxy'].split(' ')[0].split('//')[1]
                except:
                    logger.sendToLog('no manual https proxy detected', 'INFO')
                try:
                    socks_proxy = proxysource['socksProxy'].split(' ')[0].split('//')[1]
                except:
                    socks_proxy = False
            else:
                logger.sendToLog('no manual proxy detected', 'INFO')
        else:
            logger.sendToLog('no proxy file detected', 'INFO')
    elif desktop == 'GNOME':
        process = subprocess.run(['gsettings', 'get', 'org.gnome.system.proxy', 'mode'], stdout=subprocess.PIPE)
        mode = re.search('manual', process.stdout.decode('utf-8'))
        if mode is not None:
            try:
                process = subprocess.run(['gsettings', 'get', 'org.gnome.system.proxy.http', 'host'], stdout=subprocess.PIPE)
                proxy['http_proxy_ip'] = re.search("\\'([\\w0-9\\.]+)\\'", process.stdout.decode('utf-8')).group(1)
                process = subprocess.run(['gsettings', 'get', 'org.gnome.system.proxy.http', 'port'], stdout=subprocess.PIPE)
                proxy['http_proxy_port'] = process.stdout.decode('utf-8')
            except:
                logger.sendToLog('no http proxy detected', 'INFO')
            try:
                process = subprocess.run(['gsettings', 'get', 'org.gnome.system.proxy.https', 'host'], stdout=subprocess.PIPE)
                proxy['https_proxy_ip'] = re.search("\\'([\\w0-9\\.]+)\\'", process.stdout.decode('utf-8')).group(1)
                process = subprocess.run(['gsettings', 'get', 'org.gnome.system.proxy.https', 'port'], stdout=subprocess.PIPE)
                proxy['https_proxy_port'] = process.stdout.decode('utf-8')
            except:
                logger.sendToLog('no https proxy detected', 'INFO')
            try:
                process = subprocess.run(['gsettings', 'get', 'org.gnome.system.proxy.ftp', 'host'], stdout=subprocess.PIPE)
                proxy['ftp_proxy_ip'] = re.search("\\'([\\w0-9\\.]+)\\'", process.stdout.decode('utf-8')).group(1)
                process = subprocess.run(['gsettings', 'get', 'org.gnome.system.proxy.ftp', 'port'], stdout=subprocess.PIPE)
                proxy['ftp_proxy_port'] = process.stdout.decode('utf-8')
            except:
                logger.sendToLog('no ftp proxy detected', 'INFO')
            try:
                process = subprocess.run(['gsettings', 'get', 'org.gnome.system.proxy.socks', 'host'], stdout=subprocess.PIPE)
                value = re.search("\\'([\\w0-9\\.]+)\\'", process.stdout.decode('utf-8')).group(1)
                socks_proxy = True
            except:
                socks_proxy = False
        else:
            logger.sendToLog('no manual proxy detected', 'INFO')
    else:
        proxysource = urllib.request.getproxies()
        try:
            proxy['http_proxy_ip'] = proxysource['http'].split(':')[1].replace('//', '')
            proxy['http_proxy_port'] = proxysource['http'].split(':')[2].replace('/', '').replace('\n', '')
        except:
            logger.sendToLog('no http proxy detected', 'INFO')
        try:
            proxy['https_proxy_ip'] = proxysource['https'].split(':')[1].replace('//', '')
            proxy['https_proxy_port'] = proxysource['https'].split(':')[2].replace('/', '').replace('\n', '')
        except:
            logger.sendToLog('no https proxy detected', 'INFO')
        try:
            proxy['ftp_proxy_ip'] = proxysource['ftp'].split(':')[1].replace('//', '')
            proxy['ftp_proxy_port'] = proxysource['ftp'].split(':')[2].replace('/', '').replace('\n', '')
        except:
            logger.sendToLog('no ftp proxy detected', 'INFO')
        try:
            if desktop == 'Unity7':
                socks_proxy = proxysource['all'].split(':')[1].replace('//', '')
            elif os_type == OS.OSX:
                validKeys = ['SOCKSEnable']
                mac_tmp_proxies_list = {}
                proxyList = subprocess.run(['scutil', '--proxy'], stdout=subprocess.PIPE)
                for line in proxyList.stdout.decode('utf-8').split('\n'):
                    words = line.split()
                    if len(words) == 3 and words[0] in validKeys:
                        mac_tmp_proxies_list[words[0]] = words[2]
                if mac_tmp_proxies_list['SOCKSEnable'] == '1':
                    socks_proxy = True
                else:
                    socks_proxy = False
            else:
                socks_proxy = proxysource['socks'].split(':')[1].replace('//', '')
        except:
            socks_proxy = False
    key_is_available = False
    key_list = ['http_proxy_ip', 'https_proxy_ip', 'ftp_proxy_ip']
    for key in key_list:
        if key in proxy.keys():
            key_is_available = True
    if not key_is_available and socks_proxy:
        socks_message = "persepolis and aria2 don't support socks\n        you must convert socks proxy to http proxy.\n        Please read this for more help:\n            https://github.com/persepolisdm/persepolis/wiki/Privoxy"
        logger.sendToLog(socks_message, 'ERROR')
    proxy_log_message = 'proxy: ' + str(proxy)
    logger.sendToLog(proxy_log_message, 'INFO')
    return proxy