import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List, Optional, Union
import trio
from streamlink.compat import is_darwin, is_win32
from streamlink.plugin.api import validate
from streamlink.session import Streamlink
from streamlink.utils.socket import find_free_port_ipv4, find_free_port_ipv6
from streamlink.webbrowser.webbrowser import Webbrowser

class ChromiumWebbrowser(Webbrowser):
    ERROR_RESOLVE = 'Could not find Chromium-based web browser executable'

    @classmethod
    def names(cls) -> List[str]:
        if False:
            while True:
                i = 10
        return ['chromium', 'chromium-browser', 'chrome', 'google-chrome', 'google-chrome-stable']

    @classmethod
    def fallback_paths(cls) -> List[Union[str, Path]]:
        if False:
            return 10
        if is_win32:
            ms_edge: List[Union[str, Path]] = [str(Path(base) / sub / 'msedge.exe') for sub in ('Microsoft\\Edge\\Application', 'Microsoft\\Edge Beta\\Application', 'Microsoft\\Edge Dev\\Application') for base in [os.getenv(env) for env in ('PROGRAMFILES', 'PROGRAMFILES(X86)')] if base is not None]
            google_chrome: List[Union[str, Path]] = [str(Path(base) / sub / 'chrome.exe') for sub in ('Google\\Chrome\\Application', 'Google\\Chrome Beta\\Application', 'Google\\Chrome Canary\\Application') for base in [os.getenv(env) for env in ('PROGRAMFILES', 'PROGRAMFILES(X86)', 'LOCALAPPDATA')] if base is not None]
            return ms_edge + google_chrome
        if is_darwin:
            return ['/Applications/Chromium.app/Contents/MacOS/Chromium', str(Path.home() / 'Applications/Chromium.app/Contents/MacOS/Chromium'), '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', str(Path.home() / 'Applications/Google Chrome.app/Contents/MacOS/Google Chrome')]
        return []

    @classmethod
    def launch_args(cls) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        return ['--autoplay-policy=user-gesture-required', '--deny-permission-prompts', '--disable-background-networking', '--disable-backgrounding-occluded-windows', '--disable-breakpad', '--disable-client-side-phishing-detection', '--disable-component-extensions-with-background-pages', '--disable-component-update', '--disable-default-apps', '--disable-extensions', '--disable-features=GlobalMediaControls', '--disable-features=MediaRouter', '--disable-features=Translate', '--disable-hang-monitor', '--disable-logging', '--disable-notifications', '--disable-popup-blocking', '--disable-prompt-on-repost', '--disable-sync', '--disk-cache-size=0', '--metrics-recording-only', '--mute-audio', '--no-default-browser-check', '--no-experiments', '--no-first-run', '--no-service-autorun', '--password-store=basic', '--silent-launch', '--use-mock-keychain', '--window-size=0,0']

    def __init__(self, *args, host: Optional[str]=None, port: Optional[int]=None, headless: bool=True, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.host = host or '127.0.0.1'
        self.port = port
        if headless:
            self.arguments.append('--headless=new')

    @asynccontextmanager
    async def launch(self, timeout: Optional[float]=None) -> AsyncGenerator[trio.Nursery, None]:
        if self.port is None:
            if ':' in self.host:
                self.port = await find_free_port_ipv6(self.host)
            else:
                self.port = await find_free_port_ipv4(self.host)
        with self._create_temp_dir() as user_data_dir:
            arguments = self.arguments.copy()
            arguments.extend([f'--remote-debugging-host={self.host}', f'--remote-debugging-port={self.port}', f'--user-data-dir={user_data_dir}'])
            async with super()._launch(self.executable, arguments, timeout=timeout) as nursery:
                yield nursery
            await trio.sleep(0.5)

    def get_websocket_url(self, session: Streamlink) -> str:
        if False:
            for i in range(10):
                print('nop')
        return session.http.get(f"http://{(f'[{self.host}]' if ':' in self.host else self.host)}:{self.port}/json/version", retries=10, retry_backoff=0.25, retry_max_backoff=0.25, timeout=0.1, schema=validate.Schema(validate.parse_json(), {'webSocketDebuggerUrl': validate.url(scheme='ws')}, validate.get('webSocketDebuggerUrl')))