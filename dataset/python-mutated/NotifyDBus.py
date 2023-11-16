from __future__ import absolute_import
from __future__ import print_function
import sys
from .NotifyBase import NotifyBase
from ..common import NotifyImageSize
from ..common import NotifyType
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _
NOTIFY_DBUS_SUPPORT_ENABLED = False
NOTIFY_DBUS_IMAGE_SUPPORT = False
LOOP_GLIB = None
LOOP_QT = None
try:
    from dbus import SessionBus
    from dbus import Interface
    from dbus import Byte
    from dbus import ByteArray
    from dbus import DBusException
    try:
        from dbus.mainloop.glib import DBusGMainLoop
        LOOP_GLIB = DBusGMainLoop()
    except ImportError:
        pass
    try:
        from dbus.mainloop.qt import DBusQtMainLoop
        LOOP_QT = DBusQtMainLoop(set_as_default=True)
    except ImportError:
        pass
    NOTIFY_DBUS_SUPPORT_ENABLED = LOOP_GLIB is not None or LOOP_QT is not None
    if 'gobject' in sys.modules:
        del sys.modules['gobject']
    try:
        import gi
        gi.require_version('GdkPixbuf', '2.0')
        from gi.repository import GdkPixbuf
        NOTIFY_DBUS_IMAGE_SUPPORT = True
    except (ImportError, ValueError, AttributeError):
        pass
except ImportError:
    pass
MAINLOOP_MAP = {'qt': LOOP_QT, 'kde': LOOP_QT, 'glib': LOOP_GLIB, 'dbus': LOOP_QT if LOOP_QT else LOOP_GLIB}

class DBusUrgency:
    LOW = 0
    NORMAL = 1
    HIGH = 2
DBUS_URGENCIES = {DBusUrgency.LOW: 'low', DBusUrgency.NORMAL: 'normal', DBusUrgency.HIGH: 'high'}
DBUS_URGENCY_MAP = {'l': DBusUrgency.LOW, 'm': DBusUrgency.LOW, 'n': DBusUrgency.NORMAL, 'h': DBusUrgency.HIGH, 'e': DBusUrgency.HIGH, '0': DBusUrgency.LOW, '1': DBusUrgency.NORMAL, '2': DBusUrgency.HIGH}

class NotifyDBus(NotifyBase):
    """
    A wrapper for local DBus/Qt Notifications
    """
    enabled = NOTIFY_DBUS_SUPPORT_ENABLED
    requirements = {'details': _('libdbus-1.so.x must be installed.')}
    service_name = _('DBus Notification')
    service_url = 'http://www.freedesktop.org/Software/dbus/'
    protocol = list(MAINLOOP_MAP.keys())
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_dbus'
    request_rate_per_sec = 0
    image_size = NotifyImageSize.XY_128
    message_timeout_ms = 13000
    body_max_line_count = 10
    dbus_interface = 'org.freedesktop.Notifications'
    dbus_setting_location = '/org/freedesktop/Notifications'
    templates = ('{schema}://',)
    template_args = dict(NotifyBase.template_args, **{'urgency': {'name': _('Urgency'), 'type': 'choice:int', 'values': DBUS_URGENCIES, 'default': DBusUrgency.NORMAL}, 'priority': {'alias_of': 'urgency'}, 'x': {'name': _('X-Axis'), 'type': 'int', 'min': 0, 'map_to': 'x_axis'}, 'y': {'name': _('Y-Axis'), 'type': 'int', 'min': 0, 'map_to': 'y_axis'}, 'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}})

    def __init__(self, urgency=None, x_axis=None, y_axis=None, include_image=True, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize DBus Object\n        '
        super().__init__(**kwargs)
        self.registry = {}
        self.schema = kwargs.get('schema', 'dbus')
        if self.schema not in MAINLOOP_MAP:
            msg = 'The schema specified ({}) is not supported.'.format(self.schema)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.urgency = int(NotifyDBus.template_args['urgency']['default'] if urgency is None else next((v for (k, v) in DBUS_URGENCY_MAP.items() if str(urgency).lower().startswith(k)), NotifyDBus.template_args['urgency']['default']))
        if x_axis or y_axis:
            try:
                self.x_axis = int(x_axis)
                self.y_axis = int(y_axis)
            except (TypeError, ValueError):
                msg = 'The x,y coordinates specified ({},{}) are invalid.'.format(x_axis, y_axis)
                self.logger.warning(msg)
                raise TypeError(msg)
        else:
            self.x_axis = None
            self.y_axis = None
        self.include_image = include_image

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform DBus Notification\n        '
        try:
            session = SessionBus(mainloop=MAINLOOP_MAP[self.schema])
        except DBusException as e:
            self.logger.warning('Failed to send DBus notification.')
            self.logger.debug(f'DBus Exception: {e}')
            return False
        if not title:
            title = body
            body = ''
        dbus_obj = session.get_object(self.dbus_interface, self.dbus_setting_location)
        dbus_iface = Interface(dbus_obj, dbus_interface=self.dbus_interface)
        icon_path = None if not self.include_image else self.image_path(notify_type, extension='.ico')
        meta_payload = {'urgency': Byte(self.urgency)}
        if not (self.x_axis is None and self.y_axis is None):
            meta_payload['x'] = self.x_axis
            meta_payload['y'] = self.y_axis
        if NOTIFY_DBUS_IMAGE_SUPPORT and icon_path:
            try:
                image = GdkPixbuf.Pixbuf.new_from_file(icon_path)
                meta_payload['icon_data'] = (image.get_width(), image.get_height(), image.get_rowstride(), image.get_has_alpha(), image.get_bits_per_sample(), image.get_n_channels(), ByteArray(image.get_pixels()))
            except Exception as e:
                self.logger.warning('Could not load notification icon (%s).', icon_path)
                self.logger.debug(f'DBus Exception: {e}')
        try:
            self.throttle()
            dbus_iface.Notify(self.app_id, 0, '', str(title), str(body), list(), meta_payload, self.message_timeout_ms)
            self.logger.info('Sent DBus notification.')
        except Exception as e:
            self.logger.warning('Failed to send DBus notification.')
            self.logger.debug(f'DBus Exception: {e}')
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no', 'urgency': DBUS_URGENCIES[self.template_args['urgency']['default']] if self.urgency not in DBUS_URGENCIES else DBUS_URGENCIES[self.urgency]}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        if self.x_axis:
            params['x'] = str(self.x_axis)
        if self.y_axis:
            params['y'] = str(self.y_axis)
        return '{schema}://_/?{params}'.format(schema=self.schema, params=NotifyDBus.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        There are no parameters nessisary for this protocol; simply having\n        gnome:// is all you need.  This function just makes sure that\n        is in place.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        results['include_image'] = parse_bool(results['qsd'].get('image', True))
        if 'priority' in results['qsd'] and len(results['qsd']['priority']):
            results['urgency'] = NotifyDBus.unquote(results['qsd']['priority'])
        if 'urgency' in results['qsd'] and len(results['qsd']['urgency']):
            results['urgency'] = NotifyDBus.unquote(results['qsd']['urgency'])
        if 'x' in results['qsd'] and len(results['qsd']['x']):
            results['x_axis'] = NotifyDBus.unquote(results['qsd'].get('x'))
        if 'y' in results['qsd'] and len(results['qsd']['y']):
            results['y_axis'] = NotifyDBus.unquote(results['qsd'].get('y'))
        return results