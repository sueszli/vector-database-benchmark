"""
Copyright (c) 2006-2023 sqlmap developers (https://sqlmap.org/)
See the file 'LICENSE' for copying permission
"""
from lib.core.enums import PRIORITY
__priority__ = PRIORITY.NORMAL

def dependencies():
    if False:
        i = 10
        return i + 15
    pass

def tamper(payload, **kwargs):
    if False:
        return 10
    "\n    Appends a HTTP header 'X-originating-IP' to bypass Varnish Firewall\n\n    Reference:\n        * https://web.archive.org/web/20160815052159/http://community.hpe.com/t5/Protect-Your-Assets/Bypassing-web-application-firewalls-using-HTTP-headers/ba-p/6418366\n\n    Notes:\n        Examples:\n        >> X-forwarded-for: TARGET_CACHESERVER_IP (184.189.250.X)\n        >> X-remote-IP: TARGET_PROXY_IP (184.189.250.X)\n        >> X-originating-IP: TARGET_LOCAL_IP (127.0.0.1)\n        >> x-remote-addr: TARGET_INTERNALUSER_IP (192.168.1.X)\n        >> X-remote-IP: * or %00 or %0A\n    "
    headers = kwargs.get('headers', {})
    headers['X-originating-IP'] = '127.0.0.1'
    return payload