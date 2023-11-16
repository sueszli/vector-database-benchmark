"""HTTP Digest Authentication tool.

An implementation of the server-side of HTTP Digest Access
Authentication, which is described in :rfc:`2617`.

Example usage, using the built-in get_ha1_dict_plain function which uses a dict
of plaintext passwords as the credentials store::

    userpassdict = {'alice' : '4x5istwelve'}
    get_ha1 = cherrypy.lib.auth_digest.get_ha1_dict_plain(userpassdict)
    digest_auth = {'tools.auth_digest.on': True,
                   'tools.auth_digest.realm': 'wonderland',
                   'tools.auth_digest.get_ha1': get_ha1,
                   'tools.auth_digest.key': 'a565c27146791cfb',
                   'tools.auth_digest.accept_charset': 'UTF-8',
    }
    app_config = { '/' : digest_auth }
"""
import time
import functools
from hashlib import md5
from urllib.request import parse_http_list, parse_keqv_list
import cherrypy
from cherrypy._cpcompat import ntob, tonative
__author__ = 'visteya'
__date__ = 'April 2009'

def md5_hex(s):
    if False:
        while True:
            i = 10
    return md5(ntob(s, 'utf-8')).hexdigest()
qop_auth = 'auth'
qop_auth_int = 'auth-int'
valid_qops = (qop_auth, qop_auth_int)
valid_algorithms = ('MD5', 'MD5-sess')
FALLBACK_CHARSET = 'ISO-8859-1'
DEFAULT_CHARSET = 'UTF-8'

def TRACE(msg):
    if False:
        return 10
    cherrypy.log(msg, context='TOOLS.AUTH_DIGEST')

def get_ha1_dict_plain(user_password_dict):
    if False:
        i = 10
        return i + 15
    'Returns a get_ha1 function which obtains a plaintext password from a\n    dictionary of the form: {username : password}.\n\n    If you want a simple dictionary-based authentication scheme, with plaintext\n    passwords, use get_ha1_dict_plain(my_userpass_dict) as the value for the\n    get_ha1 argument to digest_auth().\n    '

    def get_ha1(realm, username):
        if False:
            for i in range(10):
                print('nop')
        password = user_password_dict.get(username)
        if password:
            return md5_hex('%s:%s:%s' % (username, realm, password))
        return None
    return get_ha1

def get_ha1_dict(user_ha1_dict):
    if False:
        for i in range(10):
            print('nop')
    'Returns a get_ha1 function which obtains a HA1 password hash from a\n    dictionary of the form: {username : HA1}.\n\n    If you want a dictionary-based authentication scheme, but with\n    pre-computed HA1 hashes instead of plain-text passwords, use\n    get_ha1_dict(my_userha1_dict) as the value for the get_ha1\n    argument to digest_auth().\n    '

    def get_ha1(realm, username):
        if False:
            return 10
        return user_ha1_dict.get(username)
    return get_ha1

def get_ha1_file_htdigest(filename):
    if False:
        return 10
    "Returns a get_ha1 function which obtains a HA1 password hash from a\n    flat file with lines of the same format as that produced by the Apache\n    htdigest utility. For example, for realm 'wonderland', username 'alice',\n    and password '4x5istwelve', the htdigest line would be::\n\n        alice:wonderland:3238cdfe91a8b2ed8e39646921a02d4c\n\n    If you want to use an Apache htdigest file as the credentials store,\n    then use get_ha1_file_htdigest(my_htdigest_file) as the value for the\n    get_ha1 argument to digest_auth().  It is recommended that the filename\n    argument be an absolute path, to avoid problems.\n    "

    def get_ha1(realm, username):
        if False:
            for i in range(10):
                print('nop')
        result = None
        with open(filename, 'r') as f:
            for line in f:
                (u, r, ha1) = line.rstrip().split(':')
                if u == username and r == realm:
                    result = ha1
                    break
        return result
    return get_ha1

def synthesize_nonce(s, key, timestamp=None):
    if False:
        i = 10
        return i + 15
    "Synthesize a nonce value which resists spoofing and can be checked\n    for staleness. Returns a string suitable as the value for 'nonce' in\n    the www-authenticate header.\n\n    s\n        A string related to the resource, such as the hostname of the server.\n\n    key\n        A secret string known only to the server.\n\n    timestamp\n        An integer seconds-since-the-epoch timestamp\n\n    "
    if timestamp is None:
        timestamp = int(time.time())
    h = md5_hex('%s:%s:%s' % (timestamp, s, key))
    nonce = '%s:%s' % (timestamp, h)
    return nonce

def H(s):
    if False:
        for i in range(10):
            print('nop')
    'The hash function H'
    return md5_hex(s)

def _try_decode_header(header, charset):
    if False:
        while True:
            i = 10
    global FALLBACK_CHARSET
    for enc in (charset, FALLBACK_CHARSET):
        try:
            return tonative(ntob(tonative(header, 'latin1'), 'latin1'), enc)
        except ValueError as ve:
            last_err = ve
    else:
        raise last_err

class HttpDigestAuthorization(object):
    """
    Parses a Digest Authorization header and performs
    re-calculation of the digest.
    """
    scheme = 'digest'

    def errmsg(self, s):
        if False:
            for i in range(10):
                print('nop')
        return 'Digest Authorization header: %s' % s

    @classmethod
    def matches(cls, header):
        if False:
            i = 10
            return i + 15
        (scheme, _, _) = header.partition(' ')
        return scheme.lower() == cls.scheme

    def __init__(self, auth_header, http_method, debug=False, accept_charset=DEFAULT_CHARSET[:]):
        if False:
            while True:
                i = 10
        self.http_method = http_method
        self.debug = debug
        if not self.matches(auth_header):
            raise ValueError('Authorization scheme is not "Digest"')
        self.auth_header = _try_decode_header(auth_header, accept_charset)
        (scheme, params) = self.auth_header.split(' ', 1)
        items = parse_http_list(params)
        paramsd = parse_keqv_list(items)
        self.realm = paramsd.get('realm')
        self.username = paramsd.get('username')
        self.nonce = paramsd.get('nonce')
        self.uri = paramsd.get('uri')
        self.method = paramsd.get('method')
        self.response = paramsd.get('response')
        self.algorithm = paramsd.get('algorithm', 'MD5').upper()
        self.cnonce = paramsd.get('cnonce')
        self.opaque = paramsd.get('opaque')
        self.qop = paramsd.get('qop')
        self.nc = paramsd.get('nc')
        if self.algorithm not in valid_algorithms:
            raise ValueError(self.errmsg("Unsupported value for algorithm: '%s'" % self.algorithm))
        has_reqd = self.username and self.realm and self.nonce and self.uri and self.response
        if not has_reqd:
            raise ValueError(self.errmsg('Not all required parameters are present.'))
        if self.qop:
            if self.qop not in valid_qops:
                raise ValueError(self.errmsg("Unsupported value for qop: '%s'" % self.qop))
            if not (self.cnonce and self.nc):
                raise ValueError(self.errmsg('If qop is sent then cnonce and nc MUST be present'))
        elif self.cnonce or self.nc:
            raise ValueError(self.errmsg('If qop is not sent, neither cnonce nor nc can be present'))

    def __str__(self):
        if False:
            print('Hello World!')
        return 'authorization : %s' % self.auth_header

    def validate_nonce(self, s, key):
        if False:
            for i in range(10):
                print('nop')
        'Validate the nonce.\n        Returns True if nonce was generated by synthesize_nonce() and the\n        timestamp is not spoofed, else returns False.\n\n        s\n            A string related to the resource, such as the hostname of\n            the server.\n\n        key\n            A secret string known only to the server.\n\n        Both s and key must be the same values which were used to synthesize\n        the nonce we are trying to validate.\n        '
        try:
            (timestamp, hashpart) = self.nonce.split(':', 1)
            (s_timestamp, s_hashpart) = synthesize_nonce(s, key, timestamp).split(':', 1)
            is_valid = s_hashpart == hashpart
            if self.debug:
                TRACE('validate_nonce: %s' % is_valid)
            return is_valid
        except ValueError:
            pass
        return False

    def is_nonce_stale(self, max_age_seconds=600):
        if False:
            print('Hello World!')
        'Returns True if a validated nonce is stale. The nonce contains a\n        timestamp in plaintext and also a secure hash of the timestamp.\n        You should first validate the nonce to ensure the plaintext\n        timestamp is not spoofed.\n        '
        try:
            (timestamp, hashpart) = self.nonce.split(':', 1)
            if int(timestamp) + max_age_seconds > int(time.time()):
                return False
        except ValueError:
            pass
        if self.debug:
            TRACE('nonce is stale')
        return True

    def HA2(self, entity_body=''):
        if False:
            print('Hello World!')
        'Returns the H(A2) string. See :rfc:`2617` section 3.2.2.3.'
        if self.qop is None or self.qop == 'auth':
            a2 = '%s:%s' % (self.http_method, self.uri)
        elif self.qop == 'auth-int':
            a2 = '%s:%s:%s' % (self.http_method, self.uri, H(entity_body))
        else:
            raise ValueError(self.errmsg('Unrecognized value for qop!'))
        return H(a2)

    def request_digest(self, ha1, entity_body=''):
        if False:
            i = 10
            return i + 15
        'Calculates the Request-Digest. See :rfc:`2617` section 3.2.2.1.\n\n        ha1\n            The HA1 string obtained from the credentials store.\n\n        entity_body\n            If \'qop\' is set to \'auth-int\', then A2 includes a hash\n            of the "entity body".  The entity body is the part of the\n            message which follows the HTTP headers. See :rfc:`2617` section\n            4.3.  This refers to the entity the user agent sent in the\n            request which has the Authorization header. Typically GET\n            requests don\'t have an entity, and POST requests do.\n\n        '
        ha2 = self.HA2(entity_body)
        if self.qop:
            req = '%s:%s:%s:%s:%s' % (self.nonce, self.nc, self.cnonce, self.qop, ha2)
        else:
            req = '%s:%s' % (self.nonce, ha2)
        if self.algorithm == 'MD5-sess':
            ha1 = H('%s:%s:%s' % (ha1, self.nonce, self.cnonce))
        digest = H('%s:%s' % (ha1, req))
        return digest

def _get_charset_declaration(charset):
    if False:
        while True:
            i = 10
    global FALLBACK_CHARSET
    charset = charset.upper()
    return ', charset="%s"' % charset if charset != FALLBACK_CHARSET else ''

def www_authenticate(realm, key, algorithm='MD5', nonce=None, qop=qop_auth, stale=False, accept_charset=DEFAULT_CHARSET[:]):
    if False:
        print('Hello World!')
    'Constructs a WWW-Authenticate header for Digest authentication.'
    if qop not in valid_qops:
        raise ValueError("Unsupported value for qop: '%s'" % qop)
    if algorithm not in valid_algorithms:
        raise ValueError("Unsupported value for algorithm: '%s'" % algorithm)
    HEADER_PATTERN = 'Digest realm="%s", nonce="%s", algorithm="%s", qop="%s"%s%s'
    if nonce is None:
        nonce = synthesize_nonce(realm, key)
    stale_param = ', stale="true"' if stale else ''
    charset_declaration = _get_charset_declaration(accept_charset)
    return HEADER_PATTERN % (realm, nonce, algorithm, qop, stale_param, charset_declaration)

def digest_auth(realm, get_ha1, key, debug=False, accept_charset='utf-8'):
    if False:
        i = 10
        return i + 15
    'A CherryPy tool that hooks at before_handler to perform\n    HTTP Digest Access Authentication, as specified in :rfc:`2617`.\n\n    If the request has an \'authorization\' header with a \'Digest\' scheme,\n    this tool authenticates the credentials supplied in that header.\n    If the request has no \'authorization\' header, or if it does but the\n    scheme is not "Digest", or if authentication fails, the tool sends\n    a 401 response with a \'WWW-Authenticate\' Digest header.\n\n    realm\n        A string containing the authentication realm.\n\n    get_ha1\n        A callable that looks up a username in a credentials store\n        and returns the HA1 string, which is defined in the RFC to be\n        MD5(username : realm : password).  The function\'s signature is:\n        ``get_ha1(realm, username)``\n        where username is obtained from the request\'s \'authorization\' header.\n        If username is not found in the credentials store, get_ha1() returns\n        None.\n\n    key\n        A secret string known only to the server, used in the synthesis\n        of nonces.\n\n    '
    request = cherrypy.serving.request
    auth_header = request.headers.get('authorization')
    respond_401 = functools.partial(_respond_401, realm, key, accept_charset, debug)
    if not HttpDigestAuthorization.matches(auth_header or ''):
        respond_401()
    msg = 'The Authorization header could not be parsed.'
    with cherrypy.HTTPError.handle(ValueError, 400, msg):
        auth = HttpDigestAuthorization(auth_header, request.method, debug=debug, accept_charset=accept_charset)
    if debug:
        TRACE(str(auth))
    if not auth.validate_nonce(realm, key):
        respond_401()
    ha1 = get_ha1(realm, auth.username)
    if ha1 is None:
        respond_401()
    digest = auth.request_digest(ha1, entity_body=request.body)
    if digest != auth.response:
        respond_401()
    if debug:
        TRACE('digest matches auth.response')
    if auth.is_nonce_stale(max_age_seconds=600):
        respond_401(stale=True)
    request.login = auth.username
    if debug:
        TRACE('authentication of %s successful' % auth.username)

def _respond_401(realm, key, accept_charset, debug, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Respond with 401 status and a WWW-Authenticate header\n    '
    header = www_authenticate(realm, key, accept_charset=accept_charset, **kwargs)
    if debug:
        TRACE(header)
    cherrypy.serving.response.headers['WWW-Authenticate'] = header
    raise cherrypy.HTTPError(401, 'You are not authorized to access that resource')