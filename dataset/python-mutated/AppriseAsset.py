import re
from uuid import uuid4
from os.path import join
from os.path import dirname
from os.path import isfile
from os.path import abspath
from .common import NotifyType
from .utils import module_detection

class AppriseAsset:
    """
    Provides a supplimentary class that can be used to provide extra
    information and details that can be used by Apprise such as providing
    an alternate location to where images/icons can be found and the
    URL masks.

    Any variable that starts with an underscore (_) can only be initialized
    by this class manually and will/can not be parsed from a configuration
    file.

    """
    app_id = 'Apprise'
    app_desc = 'Apprise Notifications'
    app_url = 'https://github.com/caronc/apprise'
    html_notify_map = {NotifyType.INFO: '#3AA3E3', NotifyType.SUCCESS: '#3AA337', NotifyType.FAILURE: '#A32037', NotifyType.WARNING: '#CACF29'}
    ascii_notify_map = {NotifyType.INFO: '[i]', NotifyType.SUCCESS: '[+]', NotifyType.FAILURE: '[!]', NotifyType.WARNING: '[~]'}
    default_html_color = '#888888'
    default_extension = '.png'
    theme = 'default'
    image_url_mask = 'https://github.com/caronc/apprise/raw/master/apprise/assets/themes/{THEME}/apprise-{TYPE}-{XY}{EXTENSION}'
    image_url_logo = 'https://github.com/caronc/apprise/raw/master/apprise/assets/themes/{THEME}/apprise-logo.png'
    image_path_mask = abspath(join(dirname(__file__), 'assets', 'themes', '{THEME}', 'apprise-{TYPE}-{XY}{EXTENSION}'))
    body_format = None
    async_mode = True
    interpret_escapes = False
    encoding = 'utf-8'
    secure_logging = True
    __plugin_paths = []
    _recursion = 0
    _uid = str(uuid4())

    def __init__(self, plugin_paths=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Asset Initialization\n\n        '
        for (key, value) in kwargs.items():
            if not hasattr(AppriseAsset, key):
                raise AttributeError('AppriseAsset init(): An invalid key {} was specified.'.format(key))
            setattr(self, key, value)
        if plugin_paths:
            module_detection(plugin_paths)

    def color(self, notify_type, color_type=None):
        if False:
            while True:
                i = 10
        "\n        Returns an HTML mapped color based on passed in notify type\n\n        if color_type is:\n           None    then a standard hex string is returned as\n                   a string format ('#000000').\n\n           int     then the integer representation is returned\n           tuple   then the the red, green, blue is returned in a tuple\n\n        "
        color = self.html_notify_map.get(notify_type, self.default_html_color)
        if color_type is None:
            return color
        elif color_type is int:
            return AppriseAsset.hex_to_int(color)
        elif color_type is tuple:
            return AppriseAsset.hex_to_rgb(color)
        raise ValueError('AppriseAsset html_color(): An invalid color_type was specified.')

    def ascii(self, notify_type):
        if False:
            i = 10
            return i + 15
        '\n        Returns an ascii representation based on passed in notify type\n\n        '
        return self.ascii_notify_map.get(notify_type, self.default_html_color)

    def image_url(self, notify_type, image_size, logo=False, extension=None):
        if False:
            while True:
                i = 10
        '\n        Apply our mask to our image URL\n\n        if logo is set to True, then the logo_url is used instead\n\n        '
        url_mask = self.image_url_logo if logo else self.image_url_mask
        if not url_mask:
            return None
        if extension is None:
            extension = self.default_extension
        re_map = {'{THEME}': self.theme if self.theme else '', '{TYPE}': notify_type, '{XY}': image_size, '{EXTENSION}': extension}
        re_table = re.compile('(' + '|'.join(re_map.keys()) + ')', re.IGNORECASE)
        return re_table.sub(lambda x: re_map[x.group()], url_mask)

    def image_path(self, notify_type, image_size, must_exist=True, extension=None):
        if False:
            return 10
        '\n        Apply our mask to our image file path\n\n        '
        if not self.image_path_mask:
            return None
        if extension is None:
            extension = self.default_extension
        re_map = {'{THEME}': self.theme if self.theme else '', '{TYPE}': notify_type, '{XY}': image_size, '{EXTENSION}': extension}
        re_table = re.compile('(' + '|'.join(re_map.keys()) + ')', re.IGNORECASE)
        path = re_table.sub(lambda x: re_map[x.group()], self.image_path_mask)
        if must_exist and (not isfile(path)):
            return None
        return path

    def image_raw(self, notify_type, image_size, extension=None):
        if False:
            return 10
        '\n        Returns the raw image if it can (otherwise the function returns None)\n\n        '
        path = self.image_path(notify_type=notify_type, image_size=image_size, extension=extension)
        if path:
            try:
                with open(path, 'rb') as fd:
                    return fd.read()
            except (OSError, IOError):
                return None
        return None

    def details(self):
        if False:
            return 10
        '\n        Returns the details associated with the AppriseAsset object\n\n        '
        return {'app_id': self.app_id, 'app_desc': self.app_desc, 'default_extension': self.default_extension, 'theme': self.theme, 'image_path_mask': self.image_path_mask, 'image_url_mask': self.image_url_mask, 'image_url_logo': self.image_url_logo}

    @staticmethod
    def hex_to_rgb(value):
        if False:
            while True:
                i = 10
        '\n        Takes a hex string (such as #00ff00) and returns a tuple in the form\n        of (red, green, blue)\n\n        eg: #00ff00 becomes : (0, 65535, 0)\n\n        '
        value = value.lstrip('#')
        lv = len(value)
        return tuple((int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)))

    @staticmethod
    def hex_to_int(value):
        if False:
            return 10
        '\n        Takes a hex string (such as #00ff00) and returns its integer\n        equivalent\n\n        eg: #00000f becomes : 15\n\n        '
        return int(value.lstrip('#'), 16)