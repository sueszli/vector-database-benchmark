import re
import os
import requests
from tempfile import NamedTemporaryFile
from .AttachBase import AttachBase
from ..common import ContentLocation
from ..URLBase import PrivacyMode
from ..AppriseLocale import gettext_lazy as _

class AttachHTTP(AttachBase):
    """
    A wrapper for HTTP based attachment sources
    """
    service_name = _('Web Based')
    protocol = 'http'
    secure_protocol = 'https'
    chunk_size = 8192
    location = ContentLocation.HOSTED

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
        self._temp_file = None
        self.qsd = {k: v for (k, v) in kwargs.get('qsd', {}).items() if k not in self.template_args}
        return

    def download(self, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform retrieval of the configuration based on the specified request\n        '
        if self.location == ContentLocation.INACCESSIBLE:
            return False
        self.invalidate()
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
        r = None
        self.throttle()
        try:
            with requests.get(url, headers=headers, auth=auth, params=self.qsd, verify=self.verify_certificate, timeout=self.request_timeout, stream=True) as r:
                r.raise_for_status()
                try:
                    file_size = int(r.headers.get('Content-Length', '0'))
                except (TypeError, ValueError):
                    file_size = 0
                if self.max_file_size > 0 and file_size > self.max_file_size:
                    self.logger.error('HTTP response exceeds allowable maximum file length ({}KB): {}'.format(int(self.max_file_size / 1024), self.url(privacy=True)))
                    return False
                self.detected_mimetype = r.headers.get('Content-Type')
                d = r.headers.get('Content-Disposition', '')
                result = re.search('filename=[\'"]?(?P<name>[^\'"]+)[\'"]?', d, re.I)
                if result:
                    self.detected_name = result.group('name').strip()
                self._temp_file = NamedTemporaryFile()
                chunk_size = self.chunk_size
                bytes_written = 0
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        self._temp_file.write(chunk)
                        bytes_written = self._temp_file.tell()
                        if self.max_file_size > 0:
                            if bytes_written > self.max_file_size:
                                self.logger.error('HTTP response exceeds allowable maximum file length ({}KB): {}'.format(int(self.max_file_size / 1024), self.url(privacy=True)))
                                self.invalidate()
                                return False
                            elif bytes_written + chunk_size > self.max_file_size:
                                self.max_file_size - bytes_written + 1
                self._temp_file.flush()
            self.download_path = self._temp_file.name
            if not self.detected_name:
                self.detected_name = os.path.basename(self.fullpath)
        except requests.RequestException as e:
            self.logger.error('A Connection error occurred retrieving HTTP configuration from %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            self.invalidate()
            return False
        except (IOError, OSError):
            self.logger.error('Could not write attachment to disk: {}'.format(self.url(privacy=True)))
            self.invalidate()
            return False
        return True

    def invalidate(self):
        if False:
            print('Hello World!')
        '\n        Close our temporary file\n        '
        if self._temp_file:
            self._temp_file.close()
            self._temp_file = None
        super().invalidate()

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        if self.cache is not None:
            if isinstance(self.cache, bool) or not self.cache:
                cache = 'yes' if self.cache else 'no'
            else:
                cache = int(self.cache)
            params['cache'] = cache
        if self._mimetype:
            params['mime'] = self._mimetype
        if self._name:
            params['name'] = self._name
        params.update({'+{}'.format(k): v for (k, v) in self.headers.items()})
        params.update(self.qsd)
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=self.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif self.user:
            auth = '{user}@'.format(user=self.quote(self.user, safe=''))
        default_port = 443 if self.secure else 80
        return '{schema}://{auth}{hostname}{port}{fullpath}?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.quote(self.host, safe=''), port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), fullpath=self.quote(self.fullpath, safe='/'), params=self.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = AttachBase.parse_url(url)
        if not results:
            return results
        results['headers'] = results['qsd-']
        results['headers'].update(results['qsd+'])
        return results