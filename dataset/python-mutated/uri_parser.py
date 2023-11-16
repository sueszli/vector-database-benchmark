"""Tools to parse and validate a MongoDB URI."""
from __future__ import annotations
import re
import sys
import warnings
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, Optional, Sized, Union, cast
from urllib.parse import unquote_plus
from pymongo.client_options import _parse_ssl_options
from pymongo.common import INTERNAL_URI_OPTION_NAME_MAP, SRV_SERVICE_NAME, URI_OPTIONS_DEPRECATION_MAP, _CaseInsensitiveDictionary, get_validated_options
from pymongo.errors import ConfigurationError, InvalidURI
from pymongo.srv_resolver import _HAVE_DNSPYTHON, _SrvResolver
from pymongo.typings import _Address
if TYPE_CHECKING:
    from pymongo.pyopenssl_context import SSLContext
SCHEME = 'mongodb://'
SCHEME_LEN = len(SCHEME)
SRV_SCHEME = 'mongodb+srv://'
SRV_SCHEME_LEN = len(SRV_SCHEME)
DEFAULT_PORT = 27017

def _unquoted_percent(s: str) -> bool:
    if False:
        print('Hello World!')
    "Check for unescaped percent signs.\n\n    :Parameters:\n        - `s`: A string. `s` can have things like '%25', '%2525',\n           and '%E2%85%A8' but cannot have unquoted percent like '%foo'.\n    "
    for i in range(len(s)):
        if s[i] == '%':
            sub = s[i:i + 3]
            if unquote_plus(sub) == sub:
                return True
    return False

def parse_userinfo(userinfo: str) -> tuple[str, str]:
    if False:
        while True:
            i = 10
    'Validates the format of user information in a MongoDB URI.\n    Reserved characters that are gen-delimiters (":", "/", "?", "#", "[",\n    "]", "@") as per RFC 3986 must be escaped.\n\n    Returns a 2-tuple containing the unescaped username followed\n    by the unescaped password.\n\n    :Parameters:\n        - `userinfo`: A string of the form <username>:<password>\n    '
    if '@' in userinfo or userinfo.count(':') > 1 or _unquoted_percent(userinfo):
        raise InvalidURI('Username and password must be escaped according to RFC 3986, use urllib.parse.quote_plus')
    (user, _, passwd) = userinfo.partition(':')
    if not user:
        raise InvalidURI('The empty string is not valid username.')
    return (unquote_plus(user), unquote_plus(passwd))

def parse_ipv6_literal_host(entity: str, default_port: Optional[int]) -> tuple[str, Optional[Union[str, int]]]:
    if False:
        print('Hello World!')
    "Validates an IPv6 literal host:port string.\n\n    Returns a 2-tuple of IPv6 literal followed by port where\n    port is default_port if it wasn't specified in entity.\n\n    :Parameters:\n        - `entity`: A string that represents an IPv6 literal enclosed\n                    in braces (e.g. '[::1]' or '[::1]:27017').\n        - `default_port`: The port number to use when one wasn't\n                          specified in entity.\n    "
    if entity.find(']') == -1:
        raise ValueError("an IPv6 address literal must be enclosed in '[' and ']' according to RFC 2732.")
    i = entity.find(']:')
    if i == -1:
        return (entity[1:-1], default_port)
    return (entity[1:i], entity[i + 2:])

def parse_host(entity: str, default_port: Optional[int]=DEFAULT_PORT) -> _Address:
    if False:
        while True:
            i = 10
    "Validates a host string\n\n    Returns a 2-tuple of host followed by port where port is default_port\n    if it wasn't specified in the string.\n\n    :Parameters:\n        - `entity`: A host or host:port string where host could be a\n                    hostname or IP address.\n        - `default_port`: The port number to use when one wasn't\n                          specified in entity.\n    "
    host = entity
    port: Optional[Union[str, int]] = default_port
    if entity[0] == '[':
        (host, port) = parse_ipv6_literal_host(entity, default_port)
    elif entity.endswith('.sock'):
        return (entity, default_port)
    elif entity.find(':') != -1:
        if entity.count(':') > 1:
            raise ValueError("Reserved characters such as ':' must be escaped according RFC 2396. An IPv6 address literal must be enclosed in '[' and ']' according to RFC 2732.")
        (host, port) = host.split(':', 1)
    if isinstance(port, str):
        if not port.isdigit() or int(port) > 65535 or int(port) <= 0:
            raise ValueError(f'Port must be an integer between 0 and 65535: {port!r}')
        port = int(port)
    return (host.lower(), port)
_IMPLICIT_TLSINSECURE_OPTS = {'tlsallowinvalidcertificates', 'tlsallowinvalidhostnames', 'tlsdisableocspendpointcheck'}

def _parse_options(opts: str, delim: Optional[str]) -> _CaseInsensitiveDictionary:
    if False:
        i = 10
        return i + 15
    'Helper method for split_options which creates the options dict.\n    Also handles the creation of a list for the URI tag_sets/\n    readpreferencetags portion, and the use of a unicode options string.\n    '
    options = _CaseInsensitiveDictionary()
    for uriopt in opts.split(delim):
        (key, value) = uriopt.split('=')
        if key.lower() == 'readpreferencetags':
            options.setdefault(key, []).append(value)
        else:
            if key in options:
                warnings.warn(f"Duplicate URI option '{key}'.", stacklevel=2)
            if key.lower() == 'authmechanismproperties':
                val = value
            else:
                val = unquote_plus(value)
            options[key] = val
    return options

def _handle_security_options(options: _CaseInsensitiveDictionary) -> _CaseInsensitiveDictionary:
    if False:
        print('Hello World!')
    'Raise appropriate errors when conflicting TLS options are present in\n    the options dictionary.\n\n    :Parameters:\n        - `options`: Instance of _CaseInsensitiveDictionary containing\n          MongoDB URI options.\n    '
    tlsinsecure = options.get('tlsinsecure')
    if tlsinsecure is not None:
        for opt in _IMPLICIT_TLSINSECURE_OPTS:
            if opt in options:
                err_msg = 'URI options %s and %s cannot be specified simultaneously.'
                raise InvalidURI(err_msg % (options.cased_key('tlsinsecure'), options.cased_key(opt)))
    tlsallowinvalidcerts = options.get('tlsallowinvalidcertificates')
    if tlsallowinvalidcerts is not None:
        if 'tlsdisableocspendpointcheck' in options:
            err_msg = 'URI options %s and %s cannot be specified simultaneously.'
            raise InvalidURI(err_msg % ('tlsallowinvalidcertificates', options.cased_key('tlsdisableocspendpointcheck')))
        if tlsallowinvalidcerts is True:
            options['tlsdisableocspendpointcheck'] = True
    tlscrlfile = options.get('tlscrlfile')
    if tlscrlfile is not None:
        for opt in ('tlsinsecure', 'tlsallowinvalidcertificates', 'tlsdisableocspendpointcheck'):
            if options.get(opt) is True:
                err_msg = 'URI option %s=True cannot be specified when CRL checking is enabled.'
                raise InvalidURI(err_msg % (opt,))
    if 'ssl' in options and 'tls' in options:

        def truth_value(val: Any) -> Any:
            if False:
                print('Hello World!')
            if val in ('true', 'false'):
                return val == 'true'
            if isinstance(val, bool):
                return val
            return val
        if truth_value(options.get('ssl')) != truth_value(options.get('tls')):
            err_msg = 'Can not specify conflicting values for URI options %s and %s.'
            raise InvalidURI(err_msg % (options.cased_key('ssl'), options.cased_key('tls')))
    return options

def _handle_option_deprecations(options: _CaseInsensitiveDictionary) -> _CaseInsensitiveDictionary:
    if False:
        return 10
    'Issue appropriate warnings when deprecated options are present in the\n    options dictionary. Removes deprecated option key, value pairs if the\n    options dictionary is found to also have the renamed option.\n\n    :Parameters:\n        - `options`: Instance of _CaseInsensitiveDictionary containing\n          MongoDB URI options.\n    '
    for optname in list(options):
        if optname in URI_OPTIONS_DEPRECATION_MAP:
            (mode, message) = URI_OPTIONS_DEPRECATION_MAP[optname]
            if mode == 'renamed':
                newoptname = message
                if newoptname in options:
                    warn_msg = "Deprecated option '%s' ignored in favor of '%s'."
                    warnings.warn(warn_msg % (options.cased_key(optname), options.cased_key(newoptname)), DeprecationWarning, stacklevel=2)
                    options.pop(optname)
                    continue
                warn_msg = "Option '%s' is deprecated, use '%s' instead."
                warnings.warn(warn_msg % (options.cased_key(optname), newoptname), DeprecationWarning, stacklevel=2)
            elif mode == 'removed':
                warn_msg = "Option '%s' is deprecated. %s."
                warnings.warn(warn_msg % (options.cased_key(optname), message), DeprecationWarning, stacklevel=2)
    return options

def _normalize_options(options: _CaseInsensitiveDictionary) -> _CaseInsensitiveDictionary:
    if False:
        while True:
            i = 10
    'Normalizes option names in the options dictionary by converting them to\n    their internally-used names.\n\n    :Parameters:\n        - `options`: Instance of _CaseInsensitiveDictionary containing\n          MongoDB URI options.\n    '
    tlsinsecure = options.get('tlsinsecure')
    if tlsinsecure is not None:
        for opt in _IMPLICIT_TLSINSECURE_OPTS:
            options[opt] = tlsinsecure
    for optname in list(options):
        intname = INTERNAL_URI_OPTION_NAME_MAP.get(optname, None)
        if intname is not None:
            options[intname] = options.pop(optname)
    return options

def validate_options(opts: Mapping[str, Any], warn: bool=False) -> MutableMapping[str, Any]:
    if False:
        print('Hello World!')
    'Validates and normalizes options passed in a MongoDB URI.\n\n    Returns a new dictionary of validated and normalized options. If warn is\n    False then errors will be thrown for invalid options, otherwise they will\n    be ignored and a warning will be issued.\n\n    :Parameters:\n        - `opts`: A dict of MongoDB URI options.\n        - `warn` (optional): If ``True`` then warnings will be logged and\n          invalid options will be ignored. Otherwise invalid options will\n          cause errors.\n    '
    return get_validated_options(opts, warn)

def split_options(opts: str, validate: bool=True, warn: bool=False, normalize: bool=True) -> MutableMapping[str, Any]:
    if False:
        print('Hello World!')
    'Takes the options portion of a MongoDB URI, validates each option\n    and returns the options in a dictionary.\n\n    :Parameters:\n        - `opt`: A string representing MongoDB URI options.\n        - `validate`: If ``True`` (the default), validate and normalize all\n          options.\n        - `warn`: If ``False`` (the default), suppress all warnings raised\n          during validation of options.\n        - `normalize`: If ``True`` (the default), renames all options to their\n          internally-used names.\n    '
    and_idx = opts.find('&')
    semi_idx = opts.find(';')
    try:
        if and_idx >= 0 and semi_idx >= 0:
            raise InvalidURI("Can not mix '&' and ';' for option separators.")
        elif and_idx >= 0:
            options = _parse_options(opts, '&')
        elif semi_idx >= 0:
            options = _parse_options(opts, ';')
        elif opts.find('=') != -1:
            options = _parse_options(opts, None)
        else:
            raise ValueError
    except ValueError:
        raise InvalidURI('MongoDB URI options are key=value pairs.') from None
    options = _handle_security_options(options)
    options = _handle_option_deprecations(options)
    if normalize:
        options = _normalize_options(options)
    if validate:
        options = cast(_CaseInsensitiveDictionary, validate_options(options, warn))
        if options.get('authsource') == '':
            raise InvalidURI('the authSource database cannot be an empty string')
    return options

def split_hosts(hosts: str, default_port: Optional[int]=DEFAULT_PORT) -> list[_Address]:
    if False:
        for i in range(10):
            print('nop')
    "Takes a string of the form host1[:port],host2[:port]... and\n    splits it into (host, port) tuples. If [:port] isn't present the\n    default_port is used.\n\n    Returns a set of 2-tuples containing the host name (or IP) followed by\n    port number.\n\n    :Parameters:\n        - `hosts`: A string of the form host1[:port],host2[:port],...\n        - `default_port`: The port number to use when one wasn't specified\n          for a host.\n    "
    nodes = []
    for entity in hosts.split(','):
        if not entity:
            raise ConfigurationError('Empty host (or extra comma in host list).')
        port = default_port
        if entity.endswith('.sock'):
            port = None
        nodes.append(parse_host(entity, port))
    return nodes
_BAD_DB_CHARS = re.compile('[' + re.escape('/ "$') + ']')
_ALLOWED_TXT_OPTS = frozenset(['authsource', 'authSource', 'replicaset', 'replicaSet', 'loadbalanced', 'loadBalanced'])

def _check_options(nodes: Sized, options: Mapping[str, Any]) -> None:
    if False:
        for i in range(10):
            print('nop')
    if len(nodes) > 1 and options.get('directconnection'):
        raise ConfigurationError('Cannot specify multiple hosts with directConnection=true')
    if options.get('loadbalanced'):
        if len(nodes) > 1:
            raise ConfigurationError('Cannot specify multiple hosts with loadBalanced=true')
        if options.get('directconnection'):
            raise ConfigurationError('Cannot specify directConnection=true with loadBalanced=true')
        if options.get('replicaset'):
            raise ConfigurationError('Cannot specify replicaSet with loadBalanced=true')

def parse_uri(uri: str, default_port: Optional[int]=DEFAULT_PORT, validate: bool=True, warn: bool=False, normalize: bool=True, connect_timeout: Optional[float]=None, srv_service_name: Optional[str]=None, srv_max_hosts: Optional[int]=None) -> dict[str, Any]:
    if False:
        while True:
            i = 10
    'Parse and validate a MongoDB URI.\n\n    Returns a dict of the form::\n\n        {\n            \'nodelist\': <list of (host, port) tuples>,\n            \'username\': <username> or None,\n            \'password\': <password> or None,\n            \'database\': <database name> or None,\n            \'collection\': <collection name> or None,\n            \'options\': <dict of MongoDB URI options>,\n            \'fqdn\': <fqdn of the MongoDB+SRV URI> or None\n        }\n\n    If the URI scheme is "mongodb+srv://" DNS SRV and TXT lookups will be done\n    to build nodelist and options.\n\n    :Parameters:\n        - `uri`: The MongoDB URI to parse.\n        - `default_port`: The port number to use when one wasn\'t specified\n          for a host in the URI.\n        - `validate` (optional): If ``True`` (the default), validate and\n          normalize all options. Default: ``True``.\n        - `warn` (optional): When validating, if ``True`` then will warn\n          the user then ignore any invalid options or values. If ``False``,\n          validation will error when options are unsupported or values are\n          invalid. Default: ``False``.\n        - `normalize` (optional): If ``True``, convert names of URI options\n          to their internally-used names. Default: ``True``.\n        - `connect_timeout` (optional): The maximum time in milliseconds to\n          wait for a response from the DNS server.\n        - `srv_service_name` (optional): A custom SRV service name\n\n    .. versionchanged:: 4.6\n       The delimiting slash (``/``) between hosts and connection options is now optional.\n       For example, "mongodb://example.com?tls=true" is now a valid URI.\n\n    .. versionchanged:: 4.0\n       To better follow RFC 3986, unquoted percent signs ("%") are no longer\n       supported.\n\n    .. versionchanged:: 3.9\n        Added the ``normalize`` parameter.\n\n    .. versionchanged:: 3.6\n        Added support for mongodb+srv:// URIs.\n\n    .. versionchanged:: 3.5\n        Return the original value of the ``readPreference`` MongoDB URI option\n        instead of the validated read preference mode.\n\n    .. versionchanged:: 3.1\n        ``warn`` added so invalid options can be ignored.\n    '
    if uri.startswith(SCHEME):
        is_srv = False
        scheme_free = uri[SCHEME_LEN:]
    elif uri.startswith(SRV_SCHEME):
        if not _HAVE_DNSPYTHON:
            python_path = sys.executable or 'python'
            raise ConfigurationError('The "dnspython" module must be installed to use mongodb+srv:// URIs. To fix this error install pymongo again:\n %s -m pip install pymongo>=4.3' % python_path)
        is_srv = True
        scheme_free = uri[SRV_SCHEME_LEN:]
    else:
        raise InvalidURI(f"Invalid URI scheme: URI must begin with '{SCHEME}' or '{SRV_SCHEME}'")
    if not scheme_free:
        raise InvalidURI('Must provide at least one hostname or IP.')
    user = None
    passwd = None
    dbase = None
    collection = None
    options = _CaseInsensitiveDictionary()
    (host_part, _, path_part) = scheme_free.partition('/')
    if not host_part:
        host_part = path_part
        path_part = ''
    if path_part:
        (dbase, _, opts) = path_part.partition('?')
    else:
        (host_part, _, opts) = host_part.partition('?')
    if dbase:
        dbase = unquote_plus(dbase)
        if '.' in dbase:
            (dbase, collection) = dbase.split('.', 1)
        if _BAD_DB_CHARS.search(dbase):
            raise InvalidURI('Bad database name "%s"' % dbase)
    else:
        dbase = None
    if opts:
        options.update(split_options(opts, validate, warn, normalize))
    if srv_service_name is None:
        srv_service_name = options.get('srvServiceName', SRV_SERVICE_NAME)
    if '@' in host_part:
        (userinfo, _, hosts) = host_part.rpartition('@')
        (user, passwd) = parse_userinfo(userinfo)
    else:
        hosts = host_part
    if '/' in hosts:
        raise InvalidURI("Any '/' in a unix domain socket must be percent-encoded: %s" % host_part)
    hosts = unquote_plus(hosts)
    fqdn = None
    srv_max_hosts = srv_max_hosts or options.get('srvMaxHosts')
    if is_srv:
        if options.get('directConnection'):
            raise ConfigurationError(f'Cannot specify directConnection=true with {SRV_SCHEME} URIs')
        nodes = split_hosts(hosts, default_port=None)
        if len(nodes) != 1:
            raise InvalidURI(f'{SRV_SCHEME} URIs must include one, and only one, hostname')
        (fqdn, port) = nodes[0]
        if port is not None:
            raise InvalidURI(f'{SRV_SCHEME} URIs must not include a port number')
        connect_timeout = connect_timeout or options.get('connectTimeoutMS')
        dns_resolver = _SrvResolver(fqdn, connect_timeout, srv_service_name, srv_max_hosts)
        nodes = dns_resolver.get_hosts()
        dns_options = dns_resolver.get_options()
        if dns_options:
            parsed_dns_options = split_options(dns_options, validate, warn, normalize)
            if set(parsed_dns_options) - _ALLOWED_TXT_OPTS:
                raise ConfigurationError('Only authSource, replicaSet, and loadBalanced are supported from DNS')
            for (opt, val) in parsed_dns_options.items():
                if opt not in options:
                    options[opt] = val
        if options.get('loadBalanced') and srv_max_hosts:
            raise InvalidURI('You cannot specify loadBalanced with srvMaxHosts')
        if options.get('replicaSet') and srv_max_hosts:
            raise InvalidURI('You cannot specify replicaSet with srvMaxHosts')
        if 'tls' not in options and 'ssl' not in options:
            options['tls'] = True if validate else 'true'
    elif not is_srv and options.get('srvServiceName') is not None:
        raise ConfigurationError("The srvServiceName option is only allowed with 'mongodb+srv://' URIs")
    elif not is_srv and srv_max_hosts:
        raise ConfigurationError("The srvMaxHosts option is only allowed with 'mongodb+srv://' URIs")
    else:
        nodes = split_hosts(hosts, default_port=default_port)
    _check_options(nodes, options)
    return {'nodelist': nodes, 'username': user, 'password': passwd, 'database': dbase, 'collection': collection, 'options': options, 'fqdn': fqdn}

def _parse_kms_tls_options(kms_tls_options: Optional[Mapping[str, Any]]) -> dict[str, SSLContext]:
    if False:
        i = 10
        return i + 15
    'Parse KMS TLS connection options.'
    if not kms_tls_options:
        return {}
    if not isinstance(kms_tls_options, dict):
        raise TypeError('kms_tls_options must be a dict')
    contexts = {}
    for (provider, options) in kms_tls_options.items():
        if not isinstance(options, dict):
            raise TypeError(f'kms_tls_options["{provider}"] must be a dict')
        options.setdefault('tls', True)
        opts = _CaseInsensitiveDictionary(options)
        opts = _handle_security_options(opts)
        opts = _normalize_options(opts)
        opts = cast(_CaseInsensitiveDictionary, validate_options(opts))
        (ssl_context, allow_invalid_hostnames) = _parse_ssl_options(opts)
        if ssl_context is None:
            raise ConfigurationError('TLS is required for KMS providers')
        if allow_invalid_hostnames:
            raise ConfigurationError('Insecure TLS options prohibited')
        for n in ['tlsInsecure', 'tlsAllowInvalidCertificates', 'tlsAllowInvalidHostnames', 'tlsDisableCertificateRevocationCheck']:
            if n in opts:
                raise ConfigurationError(f'Insecure TLS options prohibited: {n}')
            contexts[provider] = ssl_context
    return contexts
if __name__ == '__main__':
    import pprint
    try:
        pprint.pprint(parse_uri(sys.argv[1]))
    except InvalidURI as exc:
        print(exc)
    sys.exit(0)