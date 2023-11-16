from __future__ import absolute_import
from __future__ import print_function
import platform
import subprocess
import os
from .NotifyBase import NotifyBase
from ..common import NotifyImageSize
from ..common import NotifyType
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _
NOTIFY_MACOSX_SUPPORT_ENABLED = False
if platform.system() == 'Darwin':
    (major, minor) = platform.mac_ver()[0].split('.')[:2]
    NOTIFY_MACOSX_SUPPORT_ENABLED = int(major) > 10 or (int(major) == 10 and int(minor) >= 8)

class NotifyMacOSX(NotifyBase):
    """
    A wrapper for the MacOS X terminal-notifier tool

    Source: https://github.com/julienXX/terminal-notifier
    """
    enabled = NOTIFY_MACOSX_SUPPORT_ENABLED
    requirements = {'details': _('Only works with Mac OS X 10.8 and higher. Additionally  requires that /usr/local/bin/terminal-notifier is locally accessible.')}
    service_name = _('MacOSX Notification')
    service_url = 'https://github.com/julienXX/terminal-notifier'
    protocol = 'macosx'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_macosx'
    image_size = NotifyImageSize.XY_128
    request_rate_per_sec = 0
    body_max_line_count = 10
    notify_paths = ('/opt/homebrew/bin/terminal-notifier', '/usr/local/bin/terminal-notifier', '/usr/bin/terminal-notifier', '/bin/terminal-notifier')
    templates = ('{schema}://',)
    template_args = dict(NotifyBase.template_args, **{'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}, 'sound': {'name': _('Sound'), 'type': 'string'}, 'click': {'name': _('Open/Click URL'), 'type': 'string'}})

    def __init__(self, sound=None, include_image=True, click=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize MacOSX Object\n        '
        super().__init__(**kwargs)
        self.include_image = include_image
        self.notify_path = next((p for p in self.notify_paths if os.access(p, os.X_OK)), None)
        self.click = click
        self.sound = sound

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform MacOSX Notification\n        '
        if not (self.notify_path and os.access(self.notify_path, os.X_OK)):
            self.logger.warning("MacOSX Notifications requires one of the following to be in place: '{}'.".format("', '".join(self.notify_paths)))
            return False
        cmd = [self.notify_path, '-message', body]
        if title:
            cmd.extend(['-title', title])
        if self.click:
            cmd.extend(['-open', self.click])
        if self.sound:
            cmd.extend(['-sound', self.sound])
        image_path = None if not self.include_image else self.image_url(notify_type)
        if image_path:
            cmd.extend(['-appIcon', image_path])
        self.throttle()
        self.logger.debug('MacOSX CMD: {}'.format(' '.join(cmd)))
        output = subprocess.Popen(cmd)
        output.wait()
        if output.returncode:
            self.logger.warning('Failed to send MacOSX notification.')
            self.logger.exception('MacOSX Exception')
            return False
        self.logger.info('Sent MacOSX notification.')
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no'}
        if self.click:
            params['click'] = self.click
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        if self.sound:
            params['sound'] = self.sound
        return '{schema}://_/?{params}'.format(schema=self.protocol, params=NotifyMacOSX.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        There are no parameters nessisary for this protocol; simply having\n        gnome:// is all you need.  This function just makes sure that\n        is in place.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        results['include_image'] = parse_bool(results['qsd'].get('image', True))
        if 'click' in results['qsd'] and len(results['qsd']['click']):
            results['click'] = NotifyMacOSX.unquote(results['qsd']['click'])
        if 'sound' in results['qsd'] and len(results['qsd']['sound']):
            results['sound'] = NotifyMacOSX.unquote(results['qsd']['sound'])
        return results