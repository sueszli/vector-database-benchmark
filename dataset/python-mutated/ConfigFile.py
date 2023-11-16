import re
import os
from .ConfigBase import ConfigBase
from ..common import ConfigFormat
from ..common import ContentIncludeMode
from ..AppriseLocale import gettext_lazy as _

class ConfigFile(ConfigBase):
    """
    A wrapper for File based configuration sources
    """
    service_name = _('Local File')
    protocol = 'file'
    allow_cross_includes = ContentIncludeMode.STRICT

    def __init__(self, path, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize File Object\n\n        headers can be a dictionary of key/value pairs that you want to\n        additionally include as part of the server headers to post with\n\n        '
        super().__init__(**kwargs)
        self.path = os.path.abspath(os.path.expanduser(path))
        self.config_path = os.path.dirname(self.path)
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        if isinstance(self.cache, bool) or not self.cache:
            cache = 'yes' if self.cache else 'no'
        else:
            cache = int(self.cache)
        params = {'encoding': self.encoding, 'cache': cache}
        if self.config_format:
            params['format'] = self.config_format
        return 'file://{path}{params}'.format(path=self.quote(self.path), params='?{}'.format(self.urlencode(params)) if params else '')

    def read(self, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform retrieval of the configuration based on the specified request\n        '
        response = None
        try:
            if self.max_buffer_size > 0 and os.path.getsize(self.path) > self.max_buffer_size:
                self.logger.error('File size exceeds maximum allowable buffer length ({}KB).'.format(int(self.max_buffer_size / 1024)))
                return None
        except OSError:
            self.logger.error('File is not accessible: {}'.format(self.path))
            return None
        self.throttle()
        try:
            with open(self.path, 'rt', encoding=self.encoding) as f:
                response = f.read()
        except (ValueError, UnicodeDecodeError):
            self.logger.error('File not using expected encoding ({}) : {}'.format(self.encoding, self.path))
            return None
        except (IOError, OSError):
            self.logger.error('File can not be opened for read: {}'.format(self.path))
            return None
        if self.config_format is None and re.match('^.*\\.ya?ml\\s*$', self.path, re.I) is not None:
            self.default_config_format = ConfigFormat.YAML
        return response

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        Parses the URL so that we can handle all different file paths\n        and return it as our path object\n\n        '
        results = ConfigBase.parse_url(url, verify_host=False)
        if not results:
            return results
        match = re.match('[a-z0-9]+://(?P<path>[^?]+)(\\?.*)?', url, re.I)
        if not match:
            return None
        results['path'] = ConfigFile.unquote(match.group('path'))
        return results