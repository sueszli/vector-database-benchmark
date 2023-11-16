import requests
import socket
from urllib.request import urlopen
from urllib.error import URLError
from .log import LOG

def _get_network_tests_config():
    if False:
        print('Hello World!')
    'Get network_tests object from mycroft.configuration.'
    from mycroft.configuration import Configuration
    config = Configuration.get()
    return config.get('network_tests', {})

def connected():
    if False:
        print('Hello World!')
    'Check connection by connecting to 8.8.8.8 and if google.com is\n    reachable if this fails, Check Microsoft NCSI is used as a backup.\n\n    Returns:\n        True if internet connection can be detected\n    '
    if _connected_dns():
        return _connected_google()
    else:
        return _connected_ncsi()

def _connected_ncsi():
    if False:
        for i in range(10):
            print('nop')
    'Check internet connection by retrieving the Microsoft NCSI endpoint.\n\n    Returns:\n        True if internet connection can be detected\n    '
    config = _get_network_tests_config()
    ncsi_endpoint = config.get('ncsi_endpoint')
    expected_text = config.get('ncsi_expected_text')
    try:
        r = requests.get(ncsi_endpoint)
        if r.text == expected_text:
            return True
    except Exception:
        LOG.error('Unable to verify connection via NCSI endpoint.')
    return False

def _connected_dns(host=None, port=53, timeout=3):
    if False:
        return 10
    'Check internet connection by connecting to DNS servers\n\n    Returns:\n        True if internet connection can be detected\n    '
    config = _get_network_tests_config()
    if host is None:
        host = config.get('dns_primary')
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        return True
    except IOError:
        LOG.error('Unable to connect to primary DNS server, trying secondary...')
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            dns_secondary = config.get('dns_secondary')
            s.connect((dns_secondary, port))
            return True
        except IOError:
            LOG.error('Unable to connect to secondary DNS server.')
            return False

def _connected_google():
    if False:
        while True:
            i = 10
    'Check internet connection by connecting to www.google.com\n    Returns:\n        True if connection attempt succeeded\n    '
    connect_success = False
    config = _get_network_tests_config()
    url = config.get('web_url')
    try:
        urlopen(url, timeout=3)
    except URLError as ue:
        LOG.error('Attempt to connect to internet failed: ' + str(ue.reason))
    else:
        connect_success = True
    return connect_success