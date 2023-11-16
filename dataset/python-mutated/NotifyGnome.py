from __future__ import absolute_import
from __future__ import print_function
from .NotifyBase import NotifyBase
from ..common import NotifyImageSize
from ..common import NotifyType
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _
NOTIFY_GNOME_SUPPORT_ENABLED = False
try:
    import gi
    gi.require_version('Notify', '0.7')
    from gi.repository import Notify
    from gi.repository import GdkPixbuf
    NOTIFY_GNOME_SUPPORT_ENABLED = True
except (ImportError, ValueError, AttributeError):
    pass

class GnomeUrgency:
    LOW = 0
    NORMAL = 1
    HIGH = 2
GNOME_URGENCIES = {GnomeUrgency.LOW: 'low', GnomeUrgency.NORMAL: 'normal', GnomeUrgency.HIGH: 'high'}
GNOME_URGENCY_MAP = {'l': GnomeUrgency.LOW, 'm': GnomeUrgency.LOW, 'n': GnomeUrgency.NORMAL, 'h': GnomeUrgency.HIGH, 'e': GnomeUrgency.HIGH, '0': GnomeUrgency.LOW, '1': GnomeUrgency.NORMAL, '2': GnomeUrgency.HIGH}

class NotifyGnome(NotifyBase):
    """
    A wrapper for local Gnome Notifications
    """
    enabled = NOTIFY_GNOME_SUPPORT_ENABLED
    requirements = {'details': _('A local Gnome environment is required.')}
    service_name = _('Gnome Notification')
    service_url = 'https://www.gnome.org/'
    protocol = 'gnome'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_gnome'
    image_size = NotifyImageSize.XY_128
    request_rate_per_sec = 0
    body_max_line_count = 10
    title_maxlen = 0
    templates = ('{schema}://',)
    template_args = dict(NotifyBase.template_args, **{'urgency': {'name': _('Urgency'), 'type': 'choice:int', 'values': GNOME_URGENCIES, 'default': GnomeUrgency.NORMAL}, 'priority': {'alias_of': 'urgency'}, 'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}})

    def __init__(self, urgency=None, include_image=True, **kwargs):
        if False:
            return 10
        '\n        Initialize Gnome Object\n        '
        super().__init__(**kwargs)
        self.urgency = int(NotifyGnome.template_args['urgency']['default'] if urgency is None else next((v for (k, v) in GNOME_URGENCY_MAP.items() if str(urgency).lower().startswith(k)), NotifyGnome.template_args['urgency']['default']))
        self.include_image = include_image

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform Gnome Notification\n        '
        try:
            Notify.init(self.app_id)
            icon_path = None if not self.include_image else self.image_path(notify_type, extension='.ico')
            notification = Notify.Notification.new(body)
            notification.set_urgency(self.urgency)
            self.throttle()
            if icon_path:
                try:
                    image = GdkPixbuf.Pixbuf.new_from_file(icon_path)
                    notification.set_icon_from_pixbuf(image)
                    notification.set_image_from_pixbuf(image)
                except Exception as e:
                    self.logger.warning('Could not load notification icon (%s).', icon_path)
                    self.logger.debug(f'Gnome Exception: {e}')
            notification.show()
            self.logger.info('Sent Gnome notification.')
        except Exception as e:
            self.logger.warning('Failed to send Gnome notification.')
            self.logger.debug(f'Gnome Exception: {e}')
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no', 'urgency': GNOME_URGENCIES[self.template_args['urgency']['default']] if self.urgency not in GNOME_URGENCIES else GNOME_URGENCIES[self.urgency]}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://?{params}'.format(schema=self.protocol, params=NotifyGnome.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        There are no parameters nessisary for this protocol; simply having\n        gnome:// is all you need.  This function just makes sure that\n        is in place.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        results['include_image'] = parse_bool(results['qsd'].get('image', True))
        if 'priority' in results['qsd'] and len(results['qsd']['priority']):
            results['urgency'] = NotifyGnome.unquote(results['qsd']['priority'])
        if 'urgency' in results['qsd'] and len(results['qsd']['urgency']):
            results['urgency'] = NotifyGnome.unquote(results['qsd']['urgency'])
        return results