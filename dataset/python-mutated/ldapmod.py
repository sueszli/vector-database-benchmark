"""
Salt interface to LDAP commands

:depends:   - ldap Python module
:configuration: In order to connect to LDAP, certain configuration is required
    in the minion config on the LDAP server. The minimum configuration items
    that must be set are:

    .. code-block:: yaml

        ldap.basedn: dc=acme,dc=com (example values, adjust to suit)

    If your LDAP server requires authentication then you must also set:

    .. code-block:: yaml

        ldap.anonymous: False
        ldap.binddn: admin
        ldap.bindpw: password

    In addition, the following optional values may be set:

    .. code-block:: yaml

        ldap.server: localhost (default=localhost, see warning below)
        ldap.port: 389 (default=389, standard port)
        ldap.tls: False (default=False, no TLS)
        ldap.no_verify: False (default=False, verify TLS)
        ldap.anonymous: True (default=True, bind anonymous)
        ldap.scope: 2 (default=2, ldap.SCOPE_SUBTREE)
        ldap.attrs: [saltAttr] (default=None, return all attributes)

.. warning::

    At the moment this module only recommends connection to LDAP services
    listening on ``localhost``. This is deliberate to avoid the potentially
    dangerous situation of multiple minions sending identical update commands
    to the same LDAP server. It's easy enough to override this behavior, but
    badness may ensue - you have been warned.
"""
import logging
import time
import salt.utils.data
from salt.exceptions import CommandExecutionError
try:
    import ldap
    import ldap.modlist
    HAS_LDAP = True
except ImportError:
    HAS_LDAP = False
log = logging.getLogger(__name__)
__virtualname__ = 'ldap'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load this module if the ldap config is set\n    '
    if HAS_LDAP:
        return __virtualname__
    return (False, 'The ldapmod execution module cannot be loaded: ldap config not present.')

def _config(name, key=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return a value for 'name' from command line args then config file options.\n    Specify 'key' if the config file option is not the same as 'name'.\n    "
    if key is None:
        key = name
    if name in kwargs:
        value = kwargs[name]
    else:
        value = __salt__['config.option']('ldap.{}'.format(key))
    return salt.utils.data.decode(value, to_str=True)

def _connect(**kwargs):
    if False:
        while True:
            i = 10
    '\n    Instantiate LDAP Connection class and return an LDAP connection object\n    '
    connargs = {}
    for name in ['uri', 'server', 'port', 'tls', 'no_verify', 'binddn', 'bindpw', 'anonymous']:
        connargs[name] = _config(name, **kwargs)
    return _LDAPConnection(**connargs).ldap

def search(filter, dn=None, scope=None, attrs=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Run an arbitrary LDAP query and return the results.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'ldaphost\' ldap.search "filter=cn=myhost"\n\n    Return data:\n\n    .. code-block:: python\n\n        {\'myhost\': {\'count\': 1,\n                    \'results\': [[\'cn=myhost,ou=hosts,o=acme,c=gb\',\n                                 {\'saltKeyValue\': [\'ntpserver=ntp.acme.local\',\n                                                   \'foo=myfoo\'],\n                                  \'saltState\': [\'foo\', \'bar\']}]],\n                    \'time\': {\'human\': \'1.2ms\', \'raw\': \'0.00123\'}}}\n\n    Search and connection options can be overridden by specifying the relevant\n    option as key=value pairs, for example:\n\n    .. code-block:: bash\n\n        salt \'ldaphost\' ldap.search filter=cn=myhost dn=ou=hosts,o=acme,c=gb\n        scope=1 attrs=\'\' server=\'localhost\' port=\'7393\' tls=True bindpw=\'ssh\'\n    '
    if not dn:
        dn = _config('dn', 'basedn')
    if not scope:
        scope = _config('scope')
    if attrs == '':
        attrs = None
    elif attrs is None:
        attrs = _config('attrs')
    _ldap = _connect(**kwargs)
    start = time.time()
    log.debug('Running LDAP search with filter:%s, dn:%s, scope:%s, attrs:%s', filter, dn, scope, attrs)
    results = _ldap.search_s(dn, int(scope), filter, attrs)
    elapsed = time.time() - start
    if elapsed < 0.2:
        elapsed_h = str(round(elapsed * 1000, 1)) + 'ms'
    else:
        elapsed_h = str(round(elapsed, 2)) + 's'
    ret = {'results': results, 'count': len(results), 'time': {'human': elapsed_h, 'raw': str(round(elapsed, 5))}}
    return ret

class _LDAPConnection:
    """
    Setup an LDAP connection.
    """

    def __init__(self, uri, server, port, tls, no_verify, binddn, bindpw, anonymous):
        if False:
            print('Hello World!')
        '\n        Bind to an LDAP directory using passed credentials.\n        '
        self.uri = uri
        self.server = server
        self.port = port
        self.tls = tls
        self.binddn = binddn
        self.bindpw = bindpw
        if self.uri == '':
            self.uri = 'ldap://{}:{}'.format(self.server, self.port)
        try:
            if no_verify:
                ldap.set_option(ldap.OPT_X_TLS_REQUIRE_CERT, ldap.OPT_X_TLS_NEVER)
            self.ldap = ldap.initialize('{}'.format(self.uri))
            self.ldap.protocol_version = 3
            self.ldap.set_option(ldap.OPT_REFERRALS, 0)
            if self.tls:
                self.ldap.start_tls_s()
            if not anonymous:
                self.ldap.simple_bind_s(self.binddn, self.bindpw)
        except Exception as ldap_error:
            raise CommandExecutionError('Failed to bind to LDAP server {} as {}: {}'.format(self.uri, self.binddn, ldap_error))