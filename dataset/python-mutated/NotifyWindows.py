from __future__ import absolute_import
from __future__ import print_function
from time import sleep
from .NotifyBase import NotifyBase
from ..common import NotifyImageSize
from ..common import NotifyType
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _
NOTIFY_WINDOWS_SUPPORT_ENABLED = False
try:
    import win32api
    import win32con
    import win32gui
    NOTIFY_WINDOWS_SUPPORT_ENABLED = True
except ImportError:
    pass

class NotifyWindows(NotifyBase):
    """
    A wrapper for local Windows Notifications
    """
    enabled = NOTIFY_WINDOWS_SUPPORT_ENABLED
    requirements = {'details': _('A local Microsoft Windows environment is required.')}
    service_name = 'Windows Notification'
    protocol = 'windows'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_windows'
    request_rate_per_sec = 0
    image_size = NotifyImageSize.XY_128
    body_max_line_count = 2
    default_popup_duration_sec = 12
    templates = ('{schema}://',)
    template_args = dict(NotifyBase.template_args, **{'duration': {'name': _('Duration'), 'type': 'int', 'min': 1, 'default': 12}, 'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}})

    def __init__(self, include_image=True, duration=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize Windows Object\n        '
        super().__init__(**kwargs)
        self.duration = self.default_popup_duration_sec if not (isinstance(duration, int) and duration > 0) else duration
        self.hwnd = None
        self.include_image = include_image

    def _on_destroy(self, hwnd, msg, wparam, lparam):
        if False:
            i = 10
            return i + 15
        '\n        Destroy callback function\n        '
        nid = (self.hwnd, 0)
        win32gui.Shell_NotifyIcon(win32gui.NIM_DELETE, nid)
        win32api.PostQuitMessage(0)
        return 0

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform Windows Notification\n        '
        self.throttle()
        try:
            message_map = {win32con.WM_DESTROY: self._on_destroy}
            self.wc = win32gui.WNDCLASS()
            self.hinst = self.wc.hInstance = win32api.GetModuleHandle(None)
            self.wc.lpszClassName = str('PythonTaskbar')
            self.wc.lpfnWndProc = message_map
            self.classAtom = win32gui.RegisterClass(self.wc)
            style = win32con.WS_OVERLAPPED | win32con.WS_SYSMENU
            self.hwnd = win32gui.CreateWindow(self.classAtom, 'Taskbar', style, 0, 0, win32con.CW_USEDEFAULT, win32con.CW_USEDEFAULT, 0, 0, self.hinst, None)
            win32gui.UpdateWindow(self.hwnd)
            icon_path = None if not self.include_image else self.image_path(notify_type, extension='.ico')
            if icon_path:
                icon_flags = win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE
                try:
                    hicon = win32gui.LoadImage(self.hinst, icon_path, win32con.IMAGE_ICON, 0, 0, icon_flags)
                except Exception as e:
                    self.logger.warning('Could not load windows notification icon ({}): {}'.format(icon_path, e))
                    hicon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)
            else:
                hicon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)
            flags = win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP
            nid = (self.hwnd, 0, flags, win32con.WM_USER + 20, hicon, 'Tooltip')
            win32gui.Shell_NotifyIcon(win32gui.NIM_ADD, nid)
            win32gui.Shell_NotifyIcon(win32gui.NIM_MODIFY, (self.hwnd, 0, win32gui.NIF_INFO, win32con.WM_USER + 20, hicon, 'Balloon Tooltip', body, 200, title))
            sleep(self.duration)
            win32gui.DestroyWindow(self.hwnd)
            win32gui.UnregisterClass(self.wc.lpszClassName, None)
            self.logger.info('Sent Windows notification.')
        except Exception as e:
            self.logger.warning('Failed to send Windows notification.')
            self.logger.debug('Windows Exception: {}', str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no', 'duration': str(self.duration)}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://?{params}'.format(schema=self.protocol, params=NotifyWindows.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        There are no parameters nessisary for this protocol; simply having\n        windows:// is all you need.  This function just makes sure that\n        is in place.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        results['include_image'] = parse_bool(results['qsd'].get('image', True))
        try:
            results['duration'] = int(results['qsd'].get('duration'))
        except (TypeError, ValueError):
            pass
        return results