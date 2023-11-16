import re
from .logger import logger
from time import sleep
from datetime import datetime
from xml.sax.saxutils import escape as sax_escape
from urllib.parse import unquote as _unquote
from urllib.parse import quote as _quote
from .AppriseLocale import gettext_lazy as _
from .AppriseAsset import AppriseAsset
from .utils import urlencode
from .utils import parse_url
from .utils import parse_bool
from .utils import parse_list
from .utils import parse_phone_no
PATHSPLIT_LIST_DELIM = re.compile('[ \\t\\r\\n,\\\\/]+')

class PrivacyMode:
    Secret = '*'
    Outer = 'o'
    Tail = 't'
HTML_LOOKUP = {400: 'Bad Request - Unsupported Parameters.', 401: 'Verification Failed.', 404: 'Page not found.', 405: 'Method not allowed.', 500: 'Internal server error.', 503: 'Servers are overloaded.'}

class URLBase:
    """
    This is the base class for all URL Manipulation
    """
    service_name = None
    protocol = None
    secure_protocol = None
    request_rate_per_sec = 0
    socket_connect_timeout = 4.0
    socket_read_timeout = 4.0
    tags = set()
    verify_certificate = True
    logger = logger
    templates = ()
    template_tokens = {}
    template_args = {'verify': {'name': _('Verify SSL'), 'type': 'bool', 'default': verify_certificate, '_lookup_default': 'verify_certificate'}, 'rto': {'name': _('Socket Read Timeout'), 'type': 'float', 'default': socket_read_timeout, '_lookup_default': 'socket_read_timeout'}, 'cto': {'name': _('Socket Connect Timeout'), 'type': 'float', 'default': socket_connect_timeout, '_lookup_default': 'socket_connect_timeout'}}
    template_kwargs = {}

    def __init__(self, asset=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize some general logging and common server arguments that will\n        keep things consistent when working with the children that\n        inherit this class.\n\n        '
        self.asset = asset if isinstance(asset, AppriseAsset) else AppriseAsset()
        self.verify_certificate = parse_bool(kwargs.get('verify', True))
        self.secure = kwargs.get('secure', None)
        try:
            if not isinstance(self.secure, bool):
                self.secure = kwargs.get('schema', '')[-1].lower() == 's'
        except (TypeError, IndexError):
            self.secure = False
        self.host = URLBase.unquote(kwargs.get('host'))
        self.port = kwargs.get('port')
        if self.port:
            try:
                self.port = int(self.port)
            except (TypeError, ValueError):
                self.logger.warning('Invalid port number specified {}'.format(self.port))
                self.port = None
        self.user = kwargs.get('user')
        if self.user:
            self.user = URLBase.unquote(self.user)
        self.password = kwargs.get('password')
        if self.password:
            self.password = URLBase.unquote(self.password)
        self.fullpath = URLBase.unquote(kwargs.get('fullpath'))
        if not isinstance(self.fullpath, str) or not self.fullpath:
            self.fullpath = '/'
        if 'rto' in kwargs:
            try:
                self.socket_read_timeout = float(kwargs.get('rto'))
            except (TypeError, ValueError):
                self.logger.warning('Invalid socket read timeout (rto) was specified {}'.format(kwargs.get('rto')))
        if 'cto' in kwargs:
            try:
                self.socket_connect_timeout = float(kwargs.get('cto'))
            except (TypeError, ValueError):
                self.logger.warning('Invalid socket connect timeout (cto) was specified {}'.format(kwargs.get('cto')))
        if 'tag' in kwargs:
            self.tags = set(parse_list(kwargs.get('tag'), self.tags))
        self._last_io_datetime = None

    def throttle(self, last_io=None, wait=None):
        if False:
            while True:
                i = 10
        '\n        A common throttle control\n\n        if a wait is specified, then it will force a sleep of the\n        specified time if it is larger then the calculated throttle\n        time.\n        '
        if last_io is not None:
            self._last_io_datetime = last_io
        reference = datetime.now()
        if self._last_io_datetime is None:
            self._last_io_datetime = reference
            return
        if self.request_rate_per_sec <= 0.0 and (not wait):
            return
        elapsed = (reference - self._last_io_datetime).total_seconds()
        if wait is not None:
            self.logger.debug('Throttling forced for {}s...'.format(wait))
            sleep(wait)
        elif elapsed < self.request_rate_per_sec:
            self.logger.debug('Throttling for {}s...'.format(self.request_rate_per_sec - elapsed))
            sleep(self.request_rate_per_sec - elapsed)
        self._last_io_datetime = datetime.now()
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Assembles the URL associated with the notification based on the\n        arguments provied.\n\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=URLBase.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif self.user:
            auth = '{user}@'.format(user=URLBase.quote(self.user, safe=''))
        default_port = 443 if self.secure else 80
        return '{schema}://{auth}{hostname}{port}{fullpath}?{params}'.format(schema='https' if self.secure else 'http', auth=auth, hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), fullpath=URLBase.quote(self.fullpath, safe='/') if self.fullpath else '/', params=URLBase.urlencode(params))

    def __contains__(self, tags):
        if False:
            print('Hello World!')
        '\n        Returns true if the tag specified is associated with this notification.\n\n        tag can also be a tuple, set, and/or list\n\n        '
        if isinstance(tags, (tuple, set, list)):
            return bool(set(tags) & self.tags)
        return tags in self.tags

    def __str__(self):
        if False:
            while True:
                i = 10
        '\n        Returns the url path\n        '
        return self.url(privacy=True)

    @staticmethod
    def escape_html(html, convert_new_lines=False, whitespace=True):
        if False:
            while True:
                i = 10
        "\n        Takes html text as input and escapes it so that it won't\n        conflict with any xml/html wrapping characters.\n\n        Args:\n            html (str): The HTML code to escape\n            convert_new_lines (:obj:`bool`, optional): escape new lines (\n)\n            whitespace (:obj:`bool`, optional): escape whitespace\n\n        Returns:\n            str: The escaped html\n        "
        if not isinstance(html, str) or not html:
            return ''
        escaped = sax_escape(html, {"'": '&apos;', '"': '&quot;'})
        if whitespace:
            escaped = escaped.replace(u'\t', u'&emsp;').replace(u' ', u'&nbsp;')
        if convert_new_lines:
            return escaped.replace(u'\n', u'<br/>')
        return escaped

    @staticmethod
    def unquote(content, encoding='utf-8', errors='replace'):
        if False:
            print('Hello World!')
        "\n        Replace %xx escapes by their single-character equivalent. The optional\n        encoding and errors parameters specify how to decode percent-encoded\n        sequences.\n\n        Wrapper to Python's `unquote` while remaining compatible with both\n        Python 2 & 3 since the reference to this function changed between\n        versions.\n\n        Note: errors set to 'replace' means that invalid sequences are\n              replaced by a placeholder character.\n\n        Args:\n            content (str): The quoted URI string you wish to unquote\n            encoding (:obj:`str`, optional): encoding type\n            errors (:obj:`str`, errors): how to handle invalid character found\n                in encoded string (defined by encoding)\n\n        Returns:\n            str: The unquoted URI string\n        "
        if not content:
            return ''
        return _unquote(content, encoding=encoding, errors=errors)

    @staticmethod
    def quote(content, safe='/', encoding=None, errors=None):
        if False:
            for i in range(10):
                print('nop')
        " Replaces single character non-ascii characters and URI specific\n        ones by their %xx code.\n\n        Wrapper to Python's `quote` while remaining compatible with both\n        Python 2 & 3 since the reference to this function changed between\n        versions.\n\n        Args:\n            content (str): The URI string you wish to quote\n            safe (str): non-ascii characters and URI specific ones that you\n                        do not wish to escape (if detected). Setting this\n                        string to an empty one causes everything to be\n                        escaped.\n            encoding (:obj:`str`, optional): encoding type\n            errors (:obj:`str`, errors): how to handle invalid character found\n                in encoded string (defined by encoding)\n\n        Returns:\n            str: The quoted URI string\n        "
        if not content:
            return ''
        return _quote(content, safe=safe, encoding=encoding, errors=errors)

    @staticmethod
    def pprint(content, privacy=True, mode=PrivacyMode.Outer, quote=True, safe='/', encoding=None, errors=None):
        if False:
            print('Hello World!')
        '\n        Privacy Print is used to mainpulate the string before passing it into\n        part of the URL.  It is used to mask/hide private details such as\n        tokens, passwords, apikeys, etc from on-lookers.  If the privacy=False\n        is set, then the quote variable is the next flag checked.\n\n        Quoting is never done if the privacy flag is set to true to avoid\n        skewing the expected output.\n        '
        if not privacy:
            if quote:
                return URLBase.quote(content, safe=safe, encoding=encoding, errors=errors)
            return content
        if mode is PrivacyMode.Secret:
            return '****'
        if not isinstance(content, str) or not content:
            return ''
        if mode is PrivacyMode.Tail:
            return '...{}'.format(content[-4:])
        return '{}...{}'.format(content[0:1], content[-1:])

    @staticmethod
    def urlencode(query, doseq=False, safe='', encoding=None, errors=None):
        if False:
            i = 10
            return i + 15
        "Convert a mapping object or a sequence of two-element tuples\n\n        Wrapper to Python's `urlencode` while remaining compatible with both\n        Python 2 & 3 since the reference to this function changed between\n        versions.\n\n        The resulting string is a series of key=value pairs separated by '&'\n        characters, where both key and value are quoted using the quote()\n        function.\n\n        Note: If the dictionary entry contains an entry that is set to None\n              it is not included in the final result set. If you want to\n              pass in an empty variable, set it to an empty string.\n\n        Args:\n            query (str): The dictionary to encode\n            doseq (:obj:`bool`, optional): Handle sequences\n            safe (:obj:`str`): non-ascii characters and URI specific ones that\n                you do not wish to escape (if detected). Setting this string\n                to an empty one causes everything to be escaped.\n            encoding (:obj:`str`, optional): encoding type\n            errors (:obj:`str`, errors): how to handle invalid character found\n                in encoded string (defined by encoding)\n\n        Returns:\n            str: The escaped parameters returned as a string\n        "
        return urlencode(query, doseq=doseq, safe=safe, encoding=encoding, errors=errors)

    @staticmethod
    def split_path(path, unquote=True):
        if False:
            for i in range(10):
                print('nop')
        'Splits a URL up into a list object.\n\n        Parses a specified URL and breaks it into a list.\n\n        Args:\n            path (str): The path to split up into a list.\n            unquote (:obj:`bool`, optional): call unquote on each element\n                 added to the returned list.\n\n        Returns:\n            list: A list containing all of the elements in the path\n        '
        try:
            paths = PATHSPLIT_LIST_DELIM.split(path.lstrip('/'))
            if unquote:
                paths = [URLBase.unquote(x) for x in filter(bool, paths)]
        except AttributeError:
            paths = []
        return paths

    @staticmethod
    def parse_list(content, allow_whitespace=True, unquote=True):
        if False:
            print('Hello World!')
        "A wrapper to utils.parse_list() with unquoting support\n\n        Parses a specified set of data and breaks it into a list.\n\n        Args:\n            content (str): The path to split up into a list. If a list is\n                 provided, then it's individual entries are processed.\n\n            allow_whitespace (:obj:`bool`, optional): whitespace is to be\n                 treated as a delimiter\n\n            unquote (:obj:`bool`, optional): call unquote on each element\n                 added to the returned list.\n\n        Returns:\n            list: A unique list containing all of the elements in the path\n        "
        content = parse_list(content, allow_whitespace=allow_whitespace)
        if unquote:
            content = [URLBase.unquote(x) for x in filter(bool, content)]
        return content

    @staticmethod
    def parse_phone_no(content, unquote=True):
        if False:
            while True:
                i = 10
        "A wrapper to utils.parse_phone_no() with unquoting support\n\n        Parses a specified set of data and breaks it into a list.\n\n        Args:\n            content (str): The path to split up into a list. If a list is\n                 provided, then it's individual entries are processed.\n\n            unquote (:obj:`bool`, optional): call unquote on each element\n                 added to the returned list.\n\n        Returns:\n            list: A unique list containing all of the elements in the path\n        "
        if unquote:
            try:
                content = URLBase.unquote(content)
            except TypeError:
                return []
        content = parse_phone_no(content)
        return content

    @property
    def app_id(self):
        if False:
            return 10
        return self.asset.app_id if self.asset.app_id else ''

    @property
    def app_desc(self):
        if False:
            for i in range(10):
                print('nop')
        return self.asset.app_desc if self.asset.app_desc else ''

    @property
    def app_url(self):
        if False:
            for i in range(10):
                print('nop')
        return self.asset.app_url if self.asset.app_url else ''

    @property
    def request_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        'This is primarily used to fullfill the `timeout` keyword argument\n        that is used by requests.get() and requests.put() calls.\n        '
        return (self.socket_connect_timeout, self.socket_read_timeout)

    @property
    def request_auth(self):
        if False:
            while True:
                i = 10
        'This is primarily used to fullfill the `auth` keyword argument\n        that is used by requests.get() and requests.put() calls.\n        '
        return (self.user, self.password) if self.user else None

    @property
    def request_url(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assemble a simple URL that can be used by the requests library\n\n        '
        schema = 'https' if self.secure else 'http'
        url = '%s://%s' % (schema, self.host)
        if isinstance(self.port, int):
            url += ':%d' % self.port
        return url + self.fullpath

    def url_parameters(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Provides a default set of args to work with. This can greatly\n        simplify URL construction in the acommpanied url() function.\n\n        The following property returns a dictionary (of strings) containing\n        all of the parameters that can be set on a URL and managed through\n        this class.\n        '
        return {'rto': str(self.socket_read_timeout), 'cto': str(self.socket_connect_timeout), 'verify': 'yes' if self.verify_certificate else 'no'}

    @staticmethod
    def parse_url(url, verify_host=True, plus_to_space=False, strict_port=False):
        if False:
            while True:
                i = 10
        'Parses the URL and returns it broken apart into a dictionary.\n\n        This is very specific and customized for Apprise.\n\n\n        Args:\n            url (str): The URL you want to fully parse.\n            verify_host (:obj:`bool`, optional): a flag kept with the parsed\n                 URL which some child classes will later use to verify SSL\n                 keys (if SSL transactions take place).  Unless under very\n                 specific circumstances, it is strongly recomended that\n                 you leave this default value set to True.\n\n        Returns:\n            A dictionary is returned containing the URL fully parsed if\n            successful, otherwise None is returned.\n        '
        results = parse_url(url, default_schema='unknown', verify_host=verify_host, plus_to_space=plus_to_space, strict_port=strict_port)
        if not results:
            return results
        results['secure'] = results['schema'][-1] == 's'
        results['verify'] = True
        if 'verify' in results['qsd']:
            results['verify'] = parse_bool(results['qsd'].get('verify', True))
        if 'password' in results['qsd']:
            results['password'] = results['qsd']['password']
        if 'pass' in results['qsd']:
            results['password'] = results['qsd']['pass']
        if 'user' in results['qsd']:
            results['user'] = results['qsd']['user']
        if results['password'] is None and 'user' in results['qsd']:
            presults = parse_url(results['url'])
            if presults:
                results['password'] = presults['user']
        if 'rto' in results['qsd']:
            results['rto'] = results['qsd']['rto']
        if 'cto' in results['qsd']:
            results['cto'] = results['qsd']['cto']
        if 'port' in results['qsd']:
            results['port'] = results['qsd']['port']
        return results

    @staticmethod
    def http_response_code_lookup(code, response_mask=None):
        if False:
            for i in range(10):
                print('nop')
        "Parses the interger response code returned by a remote call from\n        a web request into it's human readable string version.\n\n        You can over-ride codes or add new ones by providing your own\n        response_mask that contains a dictionary of integer -> string mapped\n        variables\n        "
        if isinstance(response_mask, dict):
            HTML_LOOKUP.update(response_mask)
        try:
            response = HTML_LOOKUP[code]
        except KeyError:
            response = ''
        return response

    def __len__(self):
        if False:
            while True:
                i = 10
        '\n        Should be over-ridden and allows the tracking of how many targets\n        are associated with each URLBase object.\n\n        Default is always 1\n        '
        return 1

    def schemas(self):
        if False:
            return 10
        'A simple function that returns a set of all schemas associated\n        with this object based on the object.protocol and\n        object.secure_protocol\n        '
        schemas = set([])
        for key in ('protocol', 'secure_protocol'):
            schema = getattr(self, key, None)
            if isinstance(schema, str):
                schemas.add(schema)
            elif isinstance(schema, (set, list, tuple)):
                for s in schema:
                    if isinstance(s, str):
                        schemas.add(s)
        return schemas