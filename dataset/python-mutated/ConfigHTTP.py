import re
import requests
from .ConfigBase import ConfigBase
from ..common import ConfigFormat
from ..common import ContentIncludeMode
from ..URLBase import PrivacyMode
from ..AppriseLocale import gettext_lazy as _
MIME_IS_YAML = re.compile('(text|application)/(x-)?yaml', re.I)
MIME_IS_TEXT = re.compile('text/(plain|html)', re.I)

class ConfigHTTP(ConfigBase):
    """
    A wrapper for HTTP based configuration sources
    """
    service_name = _('Web Based')
    protocol = 'http'
    secure_protocol = 'https'
    max_error_buffer_size = 2048
    allow_cross_includes = ContentIncludeMode.ALWAYS

    def __init__(self, headers=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize HTTP Object\n\n        headers can be a dictionary of key/value pairs that you want to\n        additionally include as part of the server headers to post with\n\n        '
        super().__init__(**kwargs)
        self.schema = 'https' if self.secure else 'http'
        self.fullpath = kwargs.get('fullpath')
        if not isinstance(self.fullpath, str):
            self.fullpath = '/'
        self.headers = {}
        if headers:
            self.headers.update(headers)
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        if isinstance(self.cache, bool) or not self.cache:
            cache = 'yes' if self.cache else 'no'
        else:
            cache = int(self.cache)
        params = {'encoding': self.encoding, 'cache': cache}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        if self.config_format:
            params['format'] = self.config_format
        params.update({'+{}'.format(k): v for (k, v) in self.headers.items()})
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=self.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif self.user:
            auth = '{user}@'.format(user=self.quote(self.user, safe=''))
        default_port = 443 if self.secure else 80
        return '{schema}://{auth}{hostname}{port}{fullpath}/?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.quote(self.host, safe=''), port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), fullpath=self.quote(self.fullpath, safe='/'), params=self.urlencode(params))

    def read(self, **kwargs):
        if False:
            return 10
        '\n        Perform retrieval of the configuration based on the specified request\n        '
        headers = {'User-Agent': self.app_id}
        headers.update(self.headers)
        auth = None
        if self.user:
            auth = (self.user, self.password)
        url = '%s://%s' % (self.schema, self.host)
        if isinstance(self.port, int):
            url += ':%d' % self.port
        url += self.fullpath
        self.logger.debug('HTTP POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        response = None
        r = None
        self.throttle()
        try:
            with requests.post(url, headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout, stream=True) as r:
                r.raise_for_status()
                try:
                    file_size = int(r.headers.get('Content-Length', '0'))
                except (TypeError, ValueError):
                    file_size = 0
                if self.max_buffer_size > 0 and file_size > self.max_buffer_size:
                    self.logger.error('HTTP config response exceeds maximum buffer length ({}KB);'.format(int(self.max_buffer_size / 1024)))
                    return None
                response = r.text[:self.max_buffer_size + 1]
                if len(response) > self.max_buffer_size:
                    self.logger.error('HTTP config response exceeds maximum buffer length ({}KB);'.format(int(self.max_buffer_size / 1024)))
                    return None
                content_type = r.headers.get('Content-Type', 'application/octet-stream')
                if self.config_format is None and content_type:
                    if MIME_IS_YAML.match(content_type) is not None:
                        self.default_config_format = ConfigFormat.YAML
                    elif MIME_IS_TEXT.match(content_type) is not None:
                        self.default_config_format = ConfigFormat.TEXT
        except requests.RequestException as e:
            self.logger.error('A Connection error occurred retrieving HTTP configuration from %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return None
        return response

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = ConfigBase.parse_url(url)
        if not results:
            return results
        results['headers'] = results['qsd-']
        results['headers'].update(results['qsd+'])
        return results