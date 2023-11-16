import os
import time
import mimetypes
from ..URLBase import URLBase
from ..utils import parse_bool
from ..common import ContentLocation
from ..AppriseLocale import gettext_lazy as _

class AttachBase(URLBase):
    """
    This is the base class for all supported attachment types
    """
    max_detect_buffer_size = 131072
    unknown_mimetype = 'application/octet-stream'
    unknown_filename = 'apprise-attachment'
    unknown_filename_extension = '.obj'
    strict = False
    max_file_size = 1048576000
    location = ContentLocation.INACCESSIBLE
    template_args = {'cache': {'name': _('Cache Age'), 'type': 'int', 'default': 600}, 'mime': {'name': _('Forced Mime Type'), 'type': 'string'}, 'name': {'name': _('Forced File Name'), 'type': 'string'}, 'verify': {'name': _('Verify SSL'), 'type': 'bool', 'default': True}}

    def __init__(self, name=None, mimetype=None, cache=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize some general logging and common server arguments that will\n        keep things consistent when working with the configurations that\n        inherit this class.\n\n        Optionally provide a filename to over-ride name associated with the\n        actual file retrieved (from where-ever).\n\n        The mime-type is automatically detected, but you can over-ride this by\n        explicitly stating what it should be.\n\n        By default we cache our responses so that subsiquent calls does not\n        cause the content to be retrieved again.  For local file references\n        this makes no difference at all.  But for remote content, this does\n        mean more then one call can be made to retrieve the (same) data.  This\n        method can be somewhat inefficient if disabled.  Only disable caching\n        if you understand the consequences.\n\n        You can alternatively set the cache value to an int identifying the\n        number of seconds the previously retrieved can exist for before it\n        should be considered expired.\n        '
        super().__init__(**kwargs)
        if not mimetypes.inited:
            mimetypes.init()
        self._name = name
        self._mimetype = mimetype
        self.detected_mimetype = None
        self.detected_name = None
        self.download_path = None
        if cache is not None:
            try:
                self.cache = cache if isinstance(cache, bool) else int(cache)
            except (TypeError, ValueError):
                err = 'An invalid cache value ({}) was specified.'.format(cache)
                self.logger.warning(err)
                raise TypeError(err)
            if self.cache < 0:
                err = 'A negative cache value ({}) was specified.'.format(cache)
                self.logger.warning(err)
                raise TypeError(err)
        else:
            self.cache = None
        if self._mimetype:
            if next((t for t in mimetypes.types_map.values() if self._mimetype == t), None) is None:
                err = 'An invalid mime-type ({}) was specified.'.format(mimetype)
                self.logger.warning(err)
                raise TypeError(err)
        return

    @property
    def path(self):
        if False:
            return 10
        '\n        Returns the absolute path to the filename. If this is not known or\n        is know but has been considered expired (due to cache setting), then\n        content is re-retrieved prior to returning.\n        '
        if not self.exists():
            return None
        return self.download_path

    @property
    def name(self):
        if False:
            return 10
        '\n        Returns the filename\n        '
        if self._name:
            return self._name
        if not self.exists():
            return None
        if not self.detected_name:
            extension = mimetypes.guess_extension(self.mimetype)
            self.detected_name = '{}{}'.format(self.unknown_filename, extension if extension else self.unknown_filename_extension)
        return self.detected_name

    @property
    def mimetype(self):
        if False:
            return 10
        '\n        Returns mime type (if one is present).\n\n        Content is cached once determied to prevent overhead of future\n        calls.\n        '
        if self._mimetype:
            return self._mimetype
        if not self.exists():
            return None
        if not self.detected_mimetype:
            try:
                (self.detected_mimetype, _) = mimetypes.guess_type(self._name if self._name else self.detected_name, strict=self.strict)
            except TypeError:
                pass
        return self.detected_mimetype if self.detected_mimetype else self.unknown_mimetype

    def exists(self):
        if False:
            return 10
        '\n        Simply returns true if the object has downloaded and stored the\n        attachment AND the attachment has not expired.\n        '
        cache = self.template_args['cache']['default'] if self.cache is None else self.cache
        if self.download_path and os.path.isfile(self.download_path) and cache:
            if cache is True:
                return True
            try:
                age_in_sec = time.time() - os.stat(self.download_path).st_mtime
                if age_in_sec <= cache:
                    return True
            except (OSError, IOError):
                pass
        return self.download()

    def invalidate(self):
        if False:
            i = 10
            return i + 15
        '\n        Release any temporary data that may be open by child classes.\n        Externally fetched content should be automatically cleaned up when\n        this function is called.\n\n        This function should also reset the following entries to None:\n          - detected_name : Should identify a human readable filename\n          - download_path: Must contain a absolute path to content\n          - detected_mimetype: Should identify mimetype of content\n        '
        self.detected_name = None
        self.download_path = None
        self.detected_mimetype = None
        return

    def download(self):
        if False:
            while True:
                i = 10
        '\n        This function must be over-ridden by inheriting classes.\n\n        Inherited classes MUST populate:\n          - detected_name: Should identify a human readable filename\n          - download_path: Must contain a absolute path to content\n          - detected_mimetype: Should identify mimetype of content\n\n        If a download fails, you should ensure these values are set to None.\n        '
        raise NotImplementedError('download() is implimented by the child class.')

    @staticmethod
    def parse_url(url, verify_host=True, mimetype_db=None):
        if False:
            print('Hello World!')
        'Parses the URL and returns it broken apart into a dictionary.\n\n        This is very specific and customized for Apprise.\n\n        Args:\n            url (str): The URL you want to fully parse.\n            verify_host (:obj:`bool`, optional): a flag kept with the parsed\n                 URL which some child classes will later use to verify SSL\n                 keys (if SSL transactions take place).  Unless under very\n                 specific circumstances, it is strongly recomended that\n                 you leave this default value set to True.\n\n        Returns:\n            A dictionary is returned containing the URL fully parsed if\n            successful, otherwise None is returned.\n        '
        results = URLBase.parse_url(url, verify_host=verify_host)
        if not results:
            return results
        if 'mime' in results['qsd']:
            results['mimetype'] = results['qsd'].get('mime', '').strip().lower()
        if 'name' in results['qsd']:
            results['name'] = results['qsd'].get('name', '').strip().lower()
        if 'cache' in results['qsd']:
            try:
                results['cache'] = int(results['qsd']['cache'])
            except (ValueError, TypeError):
                results['cache'] = parse_bool(results['qsd']['cache'])
        return results

    def __len__(self):
        if False:
            return 10
        '\n        Returns the filesize of the attachment.\n\n        '
        return os.path.getsize(self.path) if self.path else 0

    def __bool__(self):
        if False:
            while True:
                i = 10
        "\n        Allows the Apprise object to be wrapped in an based 'if statement'.\n        True is returned if our content was downloaded correctly.\n        "
        return True if self.path else False