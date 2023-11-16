"""prompt_toolkit input hook for GTK 3
"""
from gi.repository import Gtk, GLib

def _main_quit(*args, **kwargs):
    if False:
        while True:
            i = 10
    Gtk.main_quit()
    return False

def inputhook(context):
    if False:
        i = 10
        return i + 15
    GLib.io_add_watch(context.fileno(), GLib.PRIORITY_DEFAULT, GLib.IO_IN, _main_quit)
    Gtk.main()