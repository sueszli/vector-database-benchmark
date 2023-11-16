"""
Query and modify an LDAP database (alternative interface)
=========================================================

.. versionadded:: 2016.3.0

This is an alternative to the ``ldap`` interface provided by the
:py:mod:`ldapmod <salt.modules.ldapmod>` execution module.

:depends: - ``ldap`` Python module
"""
import logging
import salt.utils.data
available_backends = set()
try:
    import ldap
    import ldap.ldapobject
    import ldap.modlist
    import ldap.sasl
    available_backends.add('ldap')
except ImportError:
    pass
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    'Only load this module if the Python ldap module is present'
    return bool(len(available_backends))

class LDAPError(Exception):
    """Base class of all LDAP exceptions raised by backends.

    This is only used for errors encountered while interacting with
    the LDAP server; usage errors (e.g., invalid backend name) will
    have a different type.

    :ivar cause: backend exception object, if applicable
    """

    def __init__(self, message, cause=None):
        if False:
            return 10
        super().__init__(message)
        self.cause = cause

def _convert_exception(e):
    if False:
        print('Hello World!')
    'Convert an ldap backend exception to an LDAPError and raise it.'
    raise LDAPError('exception in ldap backend: {!r}'.format(e), e) from e

def _bind(l, bind=None):
    if False:
        i = 10
        return i + 15
    'Bind helper.'
    if bind is None:
        return
    method = bind.get('method', 'simple')
    if method is None:
        return
    elif method == 'simple':
        l.simple_bind_s(bind.get('dn', ''), bind.get('password', ''))
    elif method == 'sasl':
        sasl_class = getattr(ldap.sasl, bind.get('mechanism', 'EXTERNAL').lower())
        creds = bind.get('credentials', None)
        if creds is None:
            creds = {}
        auth = sasl_class(*creds.get('args', []), **creds.get('kwargs', {}))
        l.sasl_interactive_bind_s(bind.get('dn', ''), auth)
    else:
        raise ValueError('unsupported bind method "' + method + '"; supported bind methods: simple sasl')

def _format_unicode_password(pwd):
    if False:
        return 10
    'Formats a string per Microsoft AD password specifications.\n    The string must be enclosed in double quotes and UTF-16 encoded.\n    See: https://msdn.microsoft.com/en-us/library/cc223248.aspx\n\n    :param pwd:\n       The desired password as a string\n\n    :returns:\n        A unicode string\n    '
    return '"{}"'.format(pwd).encode('utf-16-le')

class _connect_ctx:

    def __init__(self, c):
        if False:
            i = 10
            return i + 15
        self.c = c

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, *exc):
        if False:
            print('Hello World!')
        pass

def connect(connect_spec=None):
    if False:
        print('Hello World!')
    'Connect and optionally bind to an LDAP server.\n\n    :param connect_spec:\n        This can be an LDAP connection object returned by a previous\n        call to :py:func:`connect` (in which case the argument is\n        simply returned), ``None`` (in which case an empty dict is\n        used), or a dict with the following keys:\n\n        * ``\'backend\'``\n            Optional; default depends on which Python LDAP modules are\n            installed.  Name of the Python LDAP module to use.  Only\n            ``\'ldap\'`` is supported at the moment.\n\n        * ``\'url\'``\n            Optional; defaults to ``\'ldapi:///\'``.  URL to the LDAP\n            server.\n\n        * ``\'bind\'``\n            Optional; defaults to ``None``.  Describes how to bind an\n            identity to the LDAP connection.  If ``None``, an\n            anonymous connection is made.  Valid keys:\n\n            * ``\'method\'``\n                Optional; defaults to ``None``.  The authentication\n                method to use.  Valid values include but are not\n                necessarily limited to ``\'simple\'``, ``\'sasl\'``, and\n                ``None``.  If ``None``, an anonymous connection is\n                made.  Available methods depend on the chosen backend.\n\n            * ``\'mechanism\'``\n                Optional; defaults to ``\'EXTERNAL\'``.  The SASL\n                mechanism to use.  Ignored unless the method is\n                ``\'sasl\'``.  Available methods depend on the chosen\n                backend and the server\'s capabilities.\n\n            * ``\'credentials\'``\n                Optional; defaults to ``None``.  An object specific to\n                the chosen SASL mechanism and backend that represents\n                the authentication credentials.  Ignored unless the\n                method is ``\'sasl\'``.\n\n                For the ``\'ldap\'`` backend, this is a dictionary.  If\n                ``None``, an empty dict is used.  Keys:\n\n                * ``\'args\'``\n                    Optional; defaults to an empty list.  A list of\n                    arguments to pass to the SASL mechanism\n                    constructor.  See the SASL mechanism constructor\n                    documentation in the ``ldap.sasl`` Python module.\n\n                * ``\'kwargs\'``\n                    Optional; defaults to an empty dict.  A dict of\n                    keyword arguments to pass to the SASL mechanism\n                    constructor.  See the SASL mechanism constructor\n                    documentation in the ``ldap.sasl`` Python module.\n\n            * ``\'dn\'``\n                Optional; defaults to an empty string.  The\n                distinguished name to bind.\n\n            * ``\'password\'``\n                Optional; defaults to an empty string.  Password for\n                binding.  Ignored if the method is ``\'sasl\'``.\n\n        * ``\'tls\'``\n            Optional; defaults to ``None``.  A backend-specific object\n            containing settings to override default TLS behavior.\n\n            For the ``\'ldap\'`` backend, this is a dictionary.  Not all\n            settings in this dictionary are supported by all versions\n            of ``python-ldap`` or the underlying TLS library.  If\n            ``None``, an empty dict is used.  Possible keys:\n\n            * ``\'starttls\'``\n                If present, initiate a TLS connection using StartTLS.\n                (The value associated with this key is ignored.)\n\n            * ``\'cacertdir\'``\n                Set the path of the directory containing CA\n                certificates.\n\n            * ``\'cacertfile\'``\n                Set the pathname of the CA certificate file.\n\n            * ``\'certfile\'``\n                Set the pathname of the certificate file.\n\n            * ``\'cipher_suite\'``\n                Set the allowed cipher suite.\n\n            * ``\'crlcheck\'``\n                Set the CRL evaluation strategy.  Valid values are\n                ``\'none\'``, ``\'peer\'``, and ``\'all\'``.\n\n            * ``\'crlfile\'``\n                Set the pathname of the CRL file.\n\n            * ``\'dhfile\'``\n                Set the pathname of the file containing the parameters\n                for Diffie-Hellman ephemeral key exchange.\n\n            * ``\'keyfile\'``\n                Set the pathname of the certificate key file.\n\n            * ``\'newctx\'``\n                If present, instruct the underlying TLS library to\n                create a new TLS context.  (The value associated with\n                this key is ignored.)\n\n            * ``\'protocol_min\'``\n                Set the minimum protocol version.\n\n            * ``\'random_file\'``\n                Set the pathname of the random file when\n                ``/dev/random`` and ``/dev/urandom`` are not\n                available.\n\n            * ``\'require_cert\'``\n                Set the certificate validation policy.  Valid values\n                are ``\'never\'``, ``\'hard\'``, ``\'demand\'``,\n                ``\'allow\'``, and ``\'try\'``.\n\n        * ``\'opts\'``\n            Optional; defaults to ``None``.  A backend-specific object\n            containing options for the backend.\n\n            For the ``\'ldap\'`` backend, this is a dictionary of\n            OpenLDAP options to set.  If ``None``, an empty dict is\n            used.  Each key is a the name of an OpenLDAP option\n            constant without the ``\'LDAP_OPT_\'`` prefix, then\n            converted to lower case.\n\n    :returns:\n        an object representing an LDAP connection that can be used as\n        the ``connect_spec`` argument to any of the functions in this\n        module (to avoid the overhead of making and terminating\n        multiple connections).\n\n        This object should be used as a context manager.  It is safe\n        to nest ``with`` statements.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' ldap3.connect "{\n            \'url\': \'ldaps://ldap.example.com/\',\n            \'bind\': {\n                \'method\': \'simple\',\n                \'dn\': \'cn=admin,dc=example,dc=com\',\n                \'password\': \'secret\'}\n        }"\n    '
    if isinstance(connect_spec, _connect_ctx):
        return connect_spec
    if connect_spec is None:
        connect_spec = {}
    backend_name = connect_spec.get('backend', 'ldap')
    if backend_name not in available_backends:
        raise ValueError('unsupported backend or required Python module' + ' unavailable: {}'.format(backend_name))
    url = connect_spec.get('url', 'ldapi:///')
    try:
        l = ldap.initialize(url)
        l.protocol_version = ldap.VERSION3
        tls = connect_spec.get('tls', None)
        if tls is None:
            tls = {}
        vars = {}
        for (k, v) in tls.items():
            if k in ('starttls', 'newctx'):
                vars[k] = True
            elif k in ('crlcheck', 'require_cert'):
                l.set_option(getattr(ldap, 'OPT_X_TLS_' + k.upper()), getattr(ldap, 'OPT_X_TLS_' + v.upper()))
            else:
                l.set_option(getattr(ldap, 'OPT_X_TLS_' + k.upper()), v)
        if vars.get('starttls', False):
            l.start_tls_s()
        if vars.get('newctx', False):
            l.set_option(ldap.OPT_X_TLS_NEWCTX, 0)
        l.set_option(ldap.OPT_REFERRALS, 0)
        opts = connect_spec.get('opts', None)
        if opts is None:
            opts = {}
        for (k, v) in opts.items():
            opt = getattr(ldap, 'OPT_' + k.upper())
            l.set_option(opt, v)
        _bind(l, connect_spec.get('bind', None))
    except ldap.LDAPError as e:
        _convert_exception(e)
    return _connect_ctx(l)

def search(connect_spec, base, scope='subtree', filterstr='(objectClass=*)', attrlist=None, attrsonly=0):
    if False:
        print('Hello World!')
    'Search an LDAP database.\n\n    :param connect_spec:\n        See the documentation for the ``connect_spec`` parameter for\n        :py:func:`connect`.\n\n    :param base:\n        Distinguished name of the entry at which to start the search.\n\n    :param scope:\n        One of the following:\n\n        * ``\'subtree\'``\n            Search the base and all of its descendants.\n\n        * ``\'base\'``\n            Search only the base itself.\n\n        * ``\'onelevel\'``\n            Search only the base\'s immediate children.\n\n    :param filterstr:\n        String representation of the filter to apply in the search.\n\n    :param attrlist:\n        Limit the returned attributes to those in the specified list.\n        If ``None``, all attributes of each entry are returned.\n\n    :param attrsonly:\n        If non-zero, don\'t return any attribute values.\n\n    :returns:\n        a dict of results.  The dict is empty if there are no results.\n        The dict maps each returned entry\'s distinguished name to a\n        dict that maps each of the matching attribute names to a list\n        of its values.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' ldap3.search "{\n            \'url\': \'ldaps://ldap.example.com/\',\n            \'bind\': {\n                \'method\': \'simple\',\n                \'dn\': \'cn=admin,dc=example,dc=com\',\n                \'password\': \'secret\',\n            },\n        }" "base=\'dc=example,dc=com\'"\n    '
    l = connect(connect_spec)
    scope = getattr(ldap, 'SCOPE_' + scope.upper())
    try:
        results = l.c.search_s(base, scope, filterstr, attrlist, attrsonly)
    except ldap.NO_SUCH_OBJECT:
        results = []
    except ldap.LDAPError as e:
        _convert_exception(e)
    return dict(results)

def add(connect_spec, dn, attributes):
    if False:
        print('Hello World!')
    'Add an entry to an LDAP database.\n\n    :param connect_spec:\n        See the documentation for the ``connect_spec`` parameter for\n        :py:func:`connect`.\n\n    :param dn:\n        Distinguished name of the entry.\n\n    :param attributes:\n        Non-empty dict mapping each of the new entry\'s attributes to a\n        non-empty iterable of values.\n\n    :returns:\n        ``True`` if successful, raises an exception otherwise.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' ldap3.add "{\n            \'url\': \'ldaps://ldap.example.com/\',\n            \'bind\': {\n                \'method\': \'simple\',\n                \'password\': \'secret\',\n            },\n        }" "dn=\'dc=example,dc=com\'" "attributes={\'example\': \'values\'}"\n    '
    l = connect(connect_spec)
    attributes = {attr: salt.utils.data.encode(list(vals)) for (attr, vals) in attributes.items()}
    log.info('adding entry: dn: %s attributes: %s', repr(dn), repr(attributes))
    if 'unicodePwd' in attributes:
        attributes['unicodePwd'] = [_format_unicode_password(x) for x in attributes['unicodePwd']]
    modlist = ldap.modlist.addModlist(attributes)
    try:
        l.c.add_s(dn, modlist)
    except ldap.LDAPError as e:
        _convert_exception(e)
    return True

def delete(connect_spec, dn):
    if False:
        i = 10
        return i + 15
    'Delete an entry from an LDAP database.\n\n    :param connect_spec:\n        See the documentation for the ``connect_spec`` parameter for\n        :py:func:`connect`.\n\n    :param dn:\n        Distinguished name of the entry.\n\n    :returns:\n        ``True`` if successful, raises an exception otherwise.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' ldap3.delete "{\n            \'url\': \'ldaps://ldap.example.com/\',\n            \'bind\': {\n                \'method\': \'simple\',\n                \'password\': \'secret\'}\n        }" dn=\'cn=admin,dc=example,dc=com\'\n    '
    l = connect(connect_spec)
    log.info('deleting entry: dn: %s', repr(dn))
    try:
        l.c.delete_s(dn)
    except ldap.LDAPError as e:
        _convert_exception(e)
    return True

def modify(connect_spec, dn, directives):
    if False:
        for i in range(10):
            print('nop')
    'Modify an entry in an LDAP database.\n\n    :param connect_spec:\n        See the documentation for the ``connect_spec`` parameter for\n        :py:func:`connect`.\n\n    :param dn:\n        Distinguished name of the entry.\n\n    :param directives:\n        Iterable of directives that indicate how to modify the entry.\n        Each directive is a tuple of the form ``(op, attr, vals)``,\n        where:\n\n        * ``op`` identifies the modification operation to perform.\n          One of:\n\n          * ``\'add\'`` to add one or more values to the attribute\n\n          * ``\'delete\'`` to delete some or all of the values from the\n            attribute.  If no values are specified with this\n            operation, all of the attribute\'s values are deleted.\n            Otherwise, only the named values are deleted.\n\n          * ``\'replace\'`` to replace all of the attribute\'s values\n            with zero or more new values\n\n        * ``attr`` names the attribute to modify\n\n        * ``vals`` is an iterable of values to add or delete\n\n    :returns:\n        ``True`` if successful, raises an exception otherwise.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' ldap3.modify "{\n            \'url\': \'ldaps://ldap.example.com/\',\n            \'bind\': {\n                \'method\': \'simple\',\n                \'password\': \'secret\'}\n        }" dn=\'cn=admin,dc=example,dc=com\'\n        directives="(\'add\', \'example\', [\'example_val\'])"\n    '
    l = connect(connect_spec)
    modlist = [(getattr(ldap, 'MOD_' + op.upper()), attr, list(vals)) for (op, attr, vals) in directives]
    for (idx, mod) in enumerate(modlist):
        if mod[1] == 'unicodePwd':
            modlist[idx] = (mod[0], mod[1], [_format_unicode_password(x) for x in mod[2]])
    modlist = salt.utils.data.decode(modlist, to_str=True, preserve_tuples=True)
    try:
        l.c.modify_s(dn, modlist)
    except ldap.LDAPError as e:
        _convert_exception(e)
    return True

def change(connect_spec, dn, before, after):
    if False:
        for i in range(10):
            print('nop')
    'Modify an entry in an LDAP database.\n\n    This does the same thing as :py:func:`modify`, but with a simpler\n    interface.  Instead of taking a list of directives, it takes a\n    before and after view of an entry, determines the differences\n    between the two, computes the directives, and executes them.\n\n    Any attribute value present in ``before`` but missing in ``after``\n    is deleted.  Any attribute value present in ``after`` but missing\n    in ``before`` is added.  Any attribute value in the database that\n    is not mentioned in either ``before`` or ``after`` is not altered.\n    Any attribute value that is present in both ``before`` and\n    ``after`` is ignored, regardless of whether that attribute value\n    exists in the database.\n\n    :param connect_spec:\n        See the documentation for the ``connect_spec`` parameter for\n        :py:func:`connect`.\n\n    :param dn:\n        Distinguished name of the entry.\n\n    :param before:\n        The expected state of the entry before modification.  This is\n        a dict mapping each attribute name to an iterable of values.\n\n    :param after:\n        The desired state of the entry after modification.  This is a\n        dict mapping each attribute name to an iterable of values.\n\n    :returns:\n        ``True`` if successful, raises an exception otherwise.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' ldap3.change "{\n            \'url\': \'ldaps://ldap.example.com/\',\n            \'bind\': {\n                \'method\': \'simple\',\n                \'password\': \'secret\'}\n        }" dn=\'cn=admin,dc=example,dc=com\'\n        before="{\'example_value\': \'before_val\'}"\n        after="{\'example_value\': \'after_val\'}"\n    '
    l = connect(connect_spec)
    before = {attr: salt.utils.data.encode(list(vals)) for (attr, vals) in before.items()}
    after = {attr: salt.utils.data.encode(list(vals)) for (attr, vals) in after.items()}
    if 'unicodePwd' in after:
        after['unicodePwd'] = [_format_unicode_password(x) for x in after['unicodePwd']]
    modlist = ldap.modlist.modifyModlist(before, after)
    try:
        l.c.modify_s(dn, modlist)
    except ldap.LDAPError as e:
        _convert_exception(e)
    return True