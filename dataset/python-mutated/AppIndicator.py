import gi
try:
    gi.require_version('XApp', '1.0')
    from gi.repository import XApp
    assert hasattr(XApp, 'StatusIcon')
    AyatanaIndicator = None
except (AssertionError, ImportError, ValueError):
    XApp = None
    try:
        gi.require_version('AppIndicator3', '0.1')
        from gi.repository import AppIndicator3
        AyatanaIndicator = AppIndicator3
    except (ImportError, ValueError):
        try:
            gi.require_version('AyatanaAppIndicator3', '0.1')
            from gi.repository import AyatanaAppIndicator3
            AyatanaIndicator = AyatanaAppIndicator3
        except (ImportError, ValueError):
            AyatanaIndicator = None
from gi.repository import Gio, Gtk

def _create_menu_item(label, command):
    if False:
        while True:
            i = 10
    menu_item = Gtk.MenuItem(label=label)
    menu_item.connect('activate', command)
    return menu_item

class AppIndicator(Gio.Application):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        if self.supports_appindicator():
            show_menu_item = _create_menu_item('Show Ulauncher', lambda *_: self.activate())
            menu = Gtk.Menu()
            menu.append(show_menu_item)
            menu.append(_create_menu_item('Preferences', lambda *_: self.show_preferences()))
            menu.append(_create_menu_item('About', lambda *_: self.show_preferences('about')))
            menu.append(Gtk.SeparatorMenuItem())
            menu.append(_create_menu_item('Exit', lambda *_: self.quit()))
            menu.show_all()
        if XApp:
            self._indicator = XApp.StatusIcon()
            self._indicator.set_icon_name('ulauncher-indicator')
            self._indicator.set_secondary_menu(menu)
            self._indicator.connect('activate', lambda *_: self.activate())
        elif AyatanaIndicator:
            self._indicator = AyatanaIndicator.Indicator.new('ulauncher', 'ulauncher-indicator', AyatanaIndicator.IndicatorCategory.APPLICATION_STATUS)
            self._indicator.set_menu(menu)
            self._indicator.set_secondary_activate_target(show_menu_item)

    def supports_appindicator(self):
        if False:
            return 10
        return bool(XApp or AyatanaIndicator)

    def toggle_appindicator(self, status=False):
        if False:
            i = 10
            return i + 15
        if XApp:
            self._indicator.set_visible(status)
        elif AyatanaIndicator:
            self._indicator.set_status(getattr(AyatanaIndicator.IndicatorStatus, 'ACTIVE' if status else 'PASSIVE'))

    def show_preferences(self, page=None):
        if False:
            return 10
        pass