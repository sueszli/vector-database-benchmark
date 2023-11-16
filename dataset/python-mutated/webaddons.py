import logging
import webbrowser
from collections.abc import Sequence
from mitmproxy import ctx

class WebAddon:

    def load(self, loader):
        if False:
            for i in range(10):
                print('nop')
        loader.add_option('web_open_browser', bool, True, 'Start a browser.')
        loader.add_option('web_debug', bool, False, 'Enable mitmweb debugging.')
        loader.add_option('web_port', int, 8081, 'Web UI port.')
        loader.add_option('web_host', str, '127.0.0.1', 'Web UI host.')
        loader.add_option('web_columns', Sequence[str], ['tls', 'icon', 'path', 'method', 'status', 'size', 'time'], 'Columns to show in the flow list')

    def running(self):
        if False:
            while True:
                i = 10
        if hasattr(ctx.options, 'web_open_browser') and ctx.options.web_open_browser:
            web_url = f'http://{ctx.options.web_host}:{ctx.options.web_port}/'
            success = open_browser(web_url)
            if not success:
                logging.info(f'No web browser found. Please open a browser and point it to {web_url}')

def open_browser(url: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Open a URL in a browser window.\n    In contrast to webbrowser.open, we limit the list of suitable browsers.\n    This gracefully degrades to a no-op on headless servers, where webbrowser.open\n    would otherwise open lynx.\n\n    Returns:\n        True, if a browser has been opened\n        False, if no suitable browser has been found.\n    '
    browsers = ('windows-default', 'macosx', 'wslview %s', 'gio', 'x-www-browser', 'gnome-open %s', 'xdg-open', 'google-chrome', 'chrome', 'chromium', 'chromium-browser', 'firefox', 'opera', 'safari')
    for browser in browsers:
        try:
            b = webbrowser.get(browser)
        except webbrowser.Error:
            pass
        else:
            if b.open(url):
                return True
    return False