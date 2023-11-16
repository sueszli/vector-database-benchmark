class BasePlugin:

    def __init__(self, src=None):
        if False:
            return 10
        self.source = src

    def authenticate(self, headers, target_host, target_port):
        if False:
            while True:
                i = 10
        pass

class AuthenticationError(Exception):

    def __init__(self, log_msg=None, response_code=403, response_headers={}, response_msg=None):
        if False:
            for i in range(10):
                print('nop')
        self.code = response_code
        self.headers = response_headers
        self.msg = response_msg
        if log_msg is None:
            log_msg = response_msg
        super().__init__('%s %s' % (self.code, log_msg))

class InvalidOriginError(AuthenticationError):

    def __init__(self, expected, actual):
        if False:
            for i in range(10):
                print('nop')
        self.expected_origin = expected
        self.actual_origin = actual
        super().__init__(response_msg='Invalid Origin', log_msg="Invalid Origin Header: Expected one of %s, got '%s'" % (expected, actual))

class BasicHTTPAuth:
    """Verifies Basic Auth headers. Specify src as username:password"""

    def __init__(self, src=None):
        if False:
            for i in range(10):
                print('nop')
        self.src = src

    def authenticate(self, headers, target_host, target_port):
        if False:
            i = 10
            return i + 15
        import base64
        auth_header = headers.get('Authorization')
        if auth_header:
            if not auth_header.startswith('Basic '):
                self.auth_error()
            try:
                user_pass_raw = base64.b64decode(auth_header[6:])
            except TypeError:
                self.auth_error()
            try:
                user_pass_as_text = user_pass_raw.decode('ISO-8859-1')
            except UnicodeDecodeError:
                self.auth_error()
            user_pass = user_pass_as_text.split(':', 1)
            if len(user_pass) != 2:
                self.auth_error()
            if not self.validate_creds(*user_pass):
                self.demand_auth()
        else:
            self.demand_auth()

    def validate_creds(self, username, password):
        if False:
            return 10
        if '%s:%s' % (username, password) == self.src:
            return True
        else:
            return False

    def auth_error(self):
        if False:
            return 10
        raise AuthenticationError(response_code=403)

    def demand_auth(self):
        if False:
            return 10
        raise AuthenticationError(response_code=401, response_headers={'WWW-Authenticate': 'Basic realm="Websockify"'})

class ExpectOrigin:

    def __init__(self, src=None):
        if False:
            for i in range(10):
                print('nop')
        if src is None:
            self.source = []
        else:
            self.source = src.split()

    def authenticate(self, headers, target_host, target_port):
        if False:
            for i in range(10):
                print('nop')
        origin = headers.get('Origin', None)
        if origin is None or origin not in self.source:
            raise InvalidOriginError(expected=self.source, actual=origin)

class ClientCertCNAuth:
    """Verifies client by SSL certificate. Specify src as whitespace separated list of common names."""

    def __init__(self, src=None):
        if False:
            while True:
                i = 10
        if src is None:
            self.source = []
        else:
            self.source = src.split()

    def authenticate(self, headers, target_host, target_port):
        if False:
            print('Hello World!')
        if headers.get('SSL_CLIENT_S_DN_CN', None) not in self.source:
            raise AuthenticationError(response_code=403)