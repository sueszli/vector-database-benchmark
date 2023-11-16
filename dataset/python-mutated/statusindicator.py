from picard import log
from picard.const.sys import IS_HAIKU, IS_MACOS, IS_WIN
DesktopStatusIndicator = None

class AbstractProgressStatusIndicator:

    def __init__(self):
        if False:
            print('Hello World!')
        self._max_pending = 0
        self._last_pending = 0

    def update(self, files=0, albums=0, pending_files=0, pending_requests=0, progress=0):
        if False:
            i = 10
            return i + 15
        if not self.is_available:
            return
        total_pending = pending_files + pending_requests
        if total_pending == self._last_pending:
            return
        previous_done = self._max_pending - self._last_pending
        self._max_pending = max(self._max_pending, previous_done + total_pending)
        self._last_pending = total_pending
        if total_pending == 0 or self._max_pending <= 1:
            self._max_pending = 0
            self.hide_progress()
            return
        self.set_progress(progress)

    @property
    def is_available(self):
        if False:
            i = 10
            return i + 15
        return True

    def hide_progress(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def set_progress(self, progress):
        if False:
            return 10
        raise NotImplementedError
if not (IS_WIN or IS_MACOS or IS_HAIKU):
    QDBusConnection = None
    try:
        from PyQt6.QtCore import QObject, pyqtSlot
        from PyQt6.QtDBus import QDBusAbstractAdaptor, QDBusConnection, QDBusMessage
    except ImportError as err:
        log.warning('Failed importing PyQt6.QtDBus: %r', err)
    else:
        from picard import PICARD_DESKTOP_NAME
        DBUS_INTERFACE = 'com.canonical.Unity.LauncherEntry'

        class UnityLauncherEntryService(QObject):

            def __init__(self, bus, app_id):
                if False:
                    i = 10
                    return i + 15
                QObject.__init__(self)
                self._bus = bus
                self._app_uri = 'application://' + app_id
                self._path = '/com/canonical/unity/launcherentry/1'
                self._progress = 0
                self._visible = False
                self._dbus_adaptor = UnityLauncherEntryAdaptor(self)
                self._available = bus.registerObject(self._path, self)

            @property
            def current_progress(self):
                if False:
                    while True:
                        i = 10
                return {'progress': self._progress, 'progress-visible': self._visible}

            @property
            def is_available(self):
                if False:
                    while True:
                        i = 10
                return self._available

            def update(self, progress, visible=True):
                if False:
                    for i in range(10):
                        print('nop')
                self._progress = progress
                self._visible = visible
                message = QDBusMessage.createSignal(self._path, DBUS_INTERFACE, 'Update')
                message.setArguments([self._app_uri, self.current_progress])
                self._bus.send(message)

            def query(self):
                if False:
                    i = 10
                    return i + 15
                return [self._app_uri, self.current_progress]

        class UnityLauncherEntryAdaptor(QDBusAbstractAdaptor):
            """ This provides the DBus adaptor to the outside world

            The supported interface is:

                <interface name="com.canonical.Unity.LauncherEntry">
                  <signal name="Update">
                    <arg direction="out" type="s" name="app_uri"/>
                    <arg direction="out" type="a{sv}" name="properties"/>
                  </signal>
                  <method name="Query">
                    <arg direction="out" type="s" name="app_uri"/>
                    <arg direction="out" type="a{sv}" name="properties"/>
                  </method>
                </interface>
            """

            def __init__(self, parent):
                if False:
                    i = 10
                    return i + 15
                super().__init__(parent)

            @pyqtSlot(name='Query', result=list)
            def query(self):
                if False:
                    print('Hello World!')
                return self.parent().query()

        class UnityLauncherEntryStatusIndicator(AbstractProgressStatusIndicator):

            def __init__(self, window):
                if False:
                    while True:
                        i = 10
                super().__init__()
                bus = QDBusConnection.sessionBus()
                self._service = UnityLauncherEntryService(bus, PICARD_DESKTOP_NAME)

            @property
            def is_available(self):
                if False:
                    print('Hello World!')
                return self._service.is_available

            def hide_progress(self):
                if False:
                    return 10
                self._service.update(0, False)

            def set_progress(self, progress):
                if False:
                    for i in range(10):
                        print('nop')
                self._service.update(progress)
        DesktopStatusIndicator = UnityLauncherEntryStatusIndicator