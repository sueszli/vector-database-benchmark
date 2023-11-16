from __future__ import annotations
import fcntl
import os
from libqtile import hook
from libqtile.log_utils import logger
from libqtile.utils import create_task
try:
    from dbus_next.aio import MessageBus
    from dbus_next.constants import BusType
    from dbus_next.errors import DBusError
    has_dbus = True
except ImportError:
    has_dbus = False
LOGIND_SERVICE = 'org.freedesktop.login1'
LOGIND_INTERFACE = LOGIND_SERVICE + '.Manager'
LOGIND_PATH = '/org/freedesktop/login1'

class Inhibitor:
    """
    Class definition to access systemd's login1 service on dbus.

    Listens for `PrepareForSleep` signals and fires appropriate hooks
    when the signal is received.

    Can also set a sleep inhibitor which will be run after the "suspend"
    hook has been fired. This helps hooked functions to complete before
    the system goes to sleep. However, the inhibitor is set to only delay
    sleep, not block it.
    """

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.bus: MessageBus | None = None
        self.sleep = False
        self.resume = False
        self.fd: int = -1

    def want_sleep(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Convenience method to set flag to show we want to know when the\n        system is going down for sleep.\n        '
        if not has_dbus:
            logger.warning('dbus-next must be installed to listen to sleep signals')
        self.sleep = True

    def want_resume(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Convenience method to set flag to show we want to know when the\n        system is waking from sleep.\n        '
        if not has_dbus:
            logger.warning('dbus-next must be installed to listen to resume signals')
        self.resume = True

    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Will create connection to dbus only if we want to listen out\n        for a sleep or wake signal.\n        '
        if not has_dbus:
            logger.warning('dbus-next is not installed. Cannot run inhibitor process.')
            return
        if not (self.sleep or self.resume):
            return
        create_task(self._start())

    async def _start(self) -> None:
        """
        Creates the bus connection and connects to the org.freedesktop.login1.Manager
        interface. Starts an inhibitor if we are listening for sleep events.
        Attaches handler to the "PrepareForSleep" signal.
        """
        self.bus = await MessageBus(bus_type=BusType.SYSTEM, negotiate_unix_fd=True).connect()
        try:
            introspection = await self.bus.introspect(LOGIND_SERVICE, LOGIND_PATH)
        except DBusError:
            logger.warning('Could not find logind service. Suspend and resume hooks will be unavailable.')
            self.bus.disconnect()
            self.bus = None
            return
        obj = self.bus.get_proxy_object(LOGIND_SERVICE, LOGIND_PATH, introspection)
        self.login = obj.get_interface(LOGIND_INTERFACE)
        if self.sleep:
            self.take()
        self.login.on_prepare_for_sleep(self.prepare_for_sleep)

    def take(self) -> None:
        if False:
            while True:
                i = 10
        'Create an inhibitor.'
        if self.fd > 0:
            self.release()
        if self.fd < 0:
            create_task(self._take())

    async def _take(self) -> None:
        """Sends the request to dbus to create an inhibitor."""
        self.fd = await self.login.call_inhibit('sleep', 'qtile', 'Run hooked functions before suspend', 'delay')
        flags = fcntl.fcntl(self.fd, fcntl.F_GETFD)
        fcntl.fcntl(self.fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)

    def release(self) -> None:
        if False:
            i = 10
            return i + 15
        'Closes the file descriptor to release the inhibitor.'
        if self.fd > 0:
            os.close(self.fd)
        else:
            logger.warning('No inhibitor available to release.')
        try:
            os.fstat(self.fd)
        except OSError:
            self.fd = -1
        else:
            logger.warning('Unable to release inhibitor.')

    def prepare_for_sleep(self, start: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Handler for "PrepareForSleep" signal.\n\n        Value of "sleep" is:\n        - True when the machine is about to sleep\n        - False when the event is over i.e. the machine has woken up\n        '
        if start:
            hook.fire('suspend')
            self.release()
        else:
            if self.sleep:
                self.take()
            hook.fire('resume')

    def stop(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Deactivates the inhibitor, removing lock and signal handler\n        before closing bus connection.\n        '
        if not has_dbus or self.bus is None:
            return
        if self.fd > 0:
            self.release()
        if self.sleep or self.resume:
            self.login.off_prepare_for_sleep(self.prepare_for_sleep)
        self.bus.disconnect()
        self.bus = None
inhibitor = Inhibitor()