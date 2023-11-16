"""
Screensaver class which watches dbus signals to see if screensaver is active
"""
import logging
import dbus
import dbus.exceptions
DBUS_SCREENSAVER_INTERFACES = ('org.cinnamon.ScreenSaver', 'org.freedesktop.ScreenSaver', 'org.gnome.ScreenSaver', 'org.mate.ScreenSaver', 'org.xfce.ScreenSaver')

class ScreensaverMonitor(object):
    """
    Simple class for monitoring signals on the Session Bus
    """

    def __init__(self, parent):
        if False:
            while True:
                i = 10
        self._logger = logging.getLogger('razer.screensaver')
        self._logger.info('Initialising DBus Screensaver Monitor')
        self._parent = parent
        self._monitoring = True
        self._active = None
        self._dbus_instances = []
        bus = dbus.SessionBus()
        for screensaver_interface in DBUS_SCREENSAVER_INTERFACES:
            bus.add_signal_receiver(self.signal_callback, dbus_interface=screensaver_interface, signal_name='ActiveChanged')

    @property
    def monitoring(self):
        if False:
            print('Hello World!')
        '\n        Monitoring property, if true then suspend/resume will be actioned.\n\n        :return: If monitoring\n        :rtype: bool\n        '
        return self._monitoring

    @monitoring.setter
    def monitoring(self, value):
        if False:
            i = 10
            return i + 15
        '\n        Monitoring property setter, if true then suspend/resume will be actioned.\n\n        :param value: If monitoring\n        :type: bool\n        '
        self._monitoring = bool(value)

    def suspend(self):
        if False:
            while True:
                i = 10
        '\n        Suspend the device\n        '
        self._logger.debug('Received screensaver active signal')
        self._parent.suspend_devices()

    def resume(self):
        if False:
            return 10
        '\n        Resume the device\n        '
        self._logger.debug('Received screensaver inactive signal')
        self._parent.resume_devices()

    def signal_callback(self, active):
        if False:
            print('Hello World!')
        '\n        Called by DBus when a signal is found\n\n        :param active: If the screensaver is active\n        :type active: dbus.Boolean\n        '
        active = bool(active)
        if self.monitoring:
            if active:
                if self._active is None or not self._active:
                    self._active = active
                    self.suspend()
            elif self._active is None or self._active:
                self._active = active
                self.resume()