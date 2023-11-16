import importlib
from django.core.exceptions import ImproperlyConfigured
from django.utils.translation import gettext as _
from cms.utils.conf import get_cms_setting

def get_request_ip_resolver():
    if False:
        print('Hello World!')
    '\n    This is the recommended method for obtaining the specified\n    CMS_REQUEST_IP_RESOLVER as it also does some basic import validation.\n\n    Returns the resolver or raises an ImproperlyConfigured exception.\n    '
    (module, attribute) = get_cms_setting('REQUEST_IP_RESOLVER').rsplit('.', 1)
    try:
        ip_resolver_module = importlib.import_module(module)
        ip_resolver = getattr(ip_resolver_module, attribute)
    except ImportError:
        raise ImproperlyConfigured(_('Unable to find the specified CMS_REQUEST_IP_RESOLVER module: "{0}".').format(module))
    except AttributeError:
        raise ImproperlyConfigured(_('Unable to find the specified CMS_REQUEST_IP_RESOLVER function: "{0}" in module "{1}".').format(attribute, module))
    return ip_resolver

def default_request_ip_resolver(request):
    if False:
        return 10
    "\n    This is a hybrid request IP resolver that attempts should address most\n    cases. Order is important here. A 'REAL_IP' header supersedes an\n    'X_FORWARDED_FOR' header which supersedes a 'REMOTE_ADDR' header.\n    "
    return real_ip(request) or x_forwarded_ip(request) or remote_addr_ip(request)

def real_ip(request):
    if False:
        print('Hello World!')
    '\n    Returns the IP Address contained in the HTTP_X_REAL_IP headers, if\n    present. Otherwise, `None`.\n\n    Should handle Nginx and some other WSGI servers.\n    '
    return request.headers.get('X-Real-Ip')

def remote_addr_ip(request):
    if False:
        i = 10
        return i + 15
    "\n    Returns the IP Address contained in the 'REMOTE_ADDR' header, if\n    present. Otherwise, `None`.\n\n    Should be suitable for local-development servers and some HTTP servers.\n    "
    return request.META.get('REMOTE_ADDR')

def x_forwarded_ip(request):
    if False:
        while True:
            i = 10
    "\n    Returns the IP Address contained in the 'HTTP_X_FORWARDED_FOR' header, if\n    present. Otherwise, `None`.\n\n    Should handle properly configured proxy servers.\n    "
    ip_address_list = request.headers.get('X-Forwarded-For')
    if ip_address_list:
        ip_address_list = ip_address_list.split(',')
        return ip_address_list[0]