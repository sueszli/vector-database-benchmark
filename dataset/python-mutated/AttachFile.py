import re
import os
from .AttachBase import AttachBase
from ..common import ContentLocation
from ..AppriseLocale import gettext_lazy as _

class AttachFile(AttachBase):
    """
    A wrapper for File based attachment sources
    """
    service_name = _('Local File')
    protocol = 'file'
    location = ContentLocation.LOCAL

    def __init__(self, path, **kwargs):
        if False:
            return 10
        '\n        Initialize Local File Attachment Object\n\n        '
        super().__init__(**kwargs)
        self.dirty_path = os.path.expanduser(path)
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {}
        if self._mimetype:
            params['mime'] = self._mimetype
        if self._name:
            params['name'] = self._name
        return 'file://{path}{params}'.format(path=self.quote(self.dirty_path), params='?{}'.format(self.urlencode(params)) if params else '')

    def download(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform retrieval of our data.\n\n        For file base attachments, our data already exists, so we only need to\n        validate it.\n        '
        if self.location == ContentLocation.INACCESSIBLE:
            return False
        self.invalidate()
        if not os.path.isfile(self.dirty_path):
            return False
        if self.max_file_size > 0 and os.path.getsize(self.dirty_path) > self.max_file_size:
            self.logger.error('Content exceeds allowable maximum file length ({}KB): {}'.format(int(self.max_file_size / 1024), self.url(privacy=True)))
            return False
        self.download_path = self.dirty_path
        self.detected_name = os.path.basename(self.download_path)
        return True

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        Parses the URL so that we can handle all different file paths\n        and return it as our path object\n\n        '
        results = AttachBase.parse_url(url, verify_host=False)
        if not results:
            return results
        match = re.match('file://(?P<path>[^?]+)(\\?.*)?', url, re.I)
        if not match:
            return None
        results['path'] = AttachFile.unquote(match.group('path'))
        return results