import logging
from gi.repository import Gdk, GdkX11, Gio
from ulauncher.utils.environment import IS_X11
logger = logging.getLogger()
if IS_X11:
    try:
        import gi
        gi.require_version('Wnck', '3.0')
        from gi.repository import Wnck
    except (ImportError, ValueError):
        logger.debug('Wnck is not installed')

def get_monitor(use_mouse_position=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    :rtype: class:Gdk.Monitor\n    '
    display = Gdk.Display.get_default()
    if use_mouse_position:
        try:
            x11_display = GdkX11.X11Display.get_default()
            seat = x11_display.get_default_seat()
            (_, x, y) = seat.get_pointer().get_position()
            return display.get_monitor_at_point(x, y)
        except Exception:
            logger.exception('Could not get monitor with X11. Defaulting to first or primary monitor')
    return display.get_primary_monitor() or display.get_monitor(0)

def get_text_scaling_factor():
    if False:
        for i in range(10):
            print('nop')
    return Gio.Settings.new('org.gnome.desktop.interface').get_double('text-scaling-factor')

def get_windows_stacked():
    if False:
        print('Hello World!')
    try:
        wnck_screen = Wnck.Screen.get_default()
        wnck_screen.force_update()
        return reversed(wnck_screen.get_windows_stacked())
    except NameError:
        return []

def get_xserver_time():
    if False:
        print('Hello World!')
    return GdkX11.x11_get_server_time(Gdk.get_default_root_window())