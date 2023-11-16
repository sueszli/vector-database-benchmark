import dbus
import openrazer.client
from gi.repository import GLib
from dbus.mainloop.glib import DBusGMainLoop
DBUS_SCREENSAVER_INTERFACES = ('org.cinnamon.ScreenSaver', 'org.freedesktop.ScreenSaver', 'org.gnome.ScreenSaver', 'org.mate.ScreenSaver', 'org.xfce.ScreenSaver')

def signal_callback(active):
    if False:
        while True:
            i = 10
    if active:
        print('Lock screen/screensaver activated')
        device.fx.static(255, 0, 0)
    else:
        print('Lock screen/screensaver deactivated')
        device.fx.static(0, 255, 0)
DBusGMainLoop(set_as_default=True)
bus = dbus.SessionBus()
for bus_name in DBUS_SCREENSAVER_INTERFACES:
    bus.add_signal_receiver(signal_callback, dbus_interface=bus_name, signal_name='ActiveChanged')
devman = openrazer.client.DeviceManager()
devman.sync_effects = True
device = devman.devices[0]
loop = GLib.MainLoop()
loop.run()