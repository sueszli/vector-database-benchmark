from gi.repository import Gio
from ulauncher.config import APP_ID
from ulauncher.modes.apps.AppResult import AppResult
from ulauncher.modes.BaseMode import BaseMode
from ulauncher.utils.Settings import Settings

class AppMode(BaseMode):

    def get_triggers(self):
        if False:
            print('Hello World!')
        settings = Settings.load()
        if not settings.enable_application_mode:
            return []
        for app in Gio.DesktopAppInfo.get_all():
            executable = app.get_executable()
            if not executable or not app.get_display_name():
                continue
            if not app.get_show_in() and (not settings.disable_desktop_filters):
                continue
            if app.get_nodisplay() and executable != 'gnome-control-center':
                continue
            if app.get_id() == f'{APP_ID}.desktop':
                continue
            yield AppResult(app)