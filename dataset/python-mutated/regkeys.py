from bottles.backend.logger import Logger
from bottles.backend.models.config import BottleConfig
from bottles.backend.models.enum import Arch
from bottles.backend.wine.catalogs import win_versions
from bottles.backend.wine.reg import Reg
from bottles.backend.wine.wineboot import WineBoot
logging = Logger()

class RegKeys:

    def __init__(self, config: BottleConfig):
        if False:
            while True:
                i = 10
        self.config = config
        self.reg = Reg(self.config)

    def set_windows(self, version: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Change Windows version in a bottle from the given\n        configuration.\n        ----------\n        supported versions:\n            - win10 (Microsoft Windows 10)\n            - win81 (Microsoft Windows 8.1)\n            - win8 (Microsoft Windows 8)\n            - win7 (Microsoft Windows 7)\n            - win2008r2 (Microsoft Windows 2008 R1)\n            - win2008 (Microsoft Windows 2008)\n            - winxp (Microsoft Windows XP)\n        ------\n        raises: ValueError\n            If the given version is invalid.\n        '
        win_version = win_versions.get(version)
        if win_version is None:
            raise ValueError('Given version is not supported.')
        if version == 'winxp' and self.config.Arch == Arch.WIN64:
            version = 'winxp64'
        wineboot = WineBoot(self.config)
        del_keys = {'HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows\\CurrentVersion': ['SubVersionNumber', 'VersionNumber'], 'HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows NT\\CurrentVersion': ['CSDVersion', 'CurrentBuildNumber', 'CurrentVersion'], 'HKEY_LOCAL_MACHINE\\System\\CurrentControlSet\\Control\\ProductOptions': 'ProductType', 'HKEY_LOCAL_MACHINE\\System\\CurrentControlSet\\Control\\ServiceCurrent': 'OS', 'HKEY_LOCAL_MACHINE\\System\\CurrentControlSet\\Control\\Windows': 'CSDVersion', 'HKEY_CURRENT_USER\\Software\\Wine': 'Version'}
        for d in del_keys:
            _val = del_keys.get(d)
            if isinstance(_val, list):
                for v in _val:
                    self.reg.remove(d, v)
            else:
                self.reg.remove(d, _val)
        if version not in ['win98', 'win95']:
            bundle = {'HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows NT\\CurrentVersion': [{'value': 'CSDVersion', 'data': win_version['CSDVersion']}, {'value': 'CurrentBuild', 'data': win_version['CurrentBuild']}, {'value': 'CurrentBuildNumber', 'data': win_version['CurrentBuildNumber']}, {'value': 'CurrentVersion', 'data': win_version['CurrentVersion']}, {'value': 'ProductName', 'data': win_version['ProductName']}, {'value': 'CurrentMinorVersionNumber', 'data': win_version['CurrentMinorVersionNumber'], 'key_type': 'dword'}, {'value': 'CurrentMajorVersionNumber', 'data': win_version['CurrentMajorVersionNumber'], 'key_type': 'dword'}], 'HKEY_LOCAL_MACHINE\\System\\CurrentControlSet\\Control\\Windows': [{'value': 'CSDVersion', 'data': win_version['CSDVersionHex'], 'key_type': 'dword'}]}
        else:
            bundle = {'HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows\\CurrentVersion': [{'value': 'ProductName', 'data': win_version['ProductName']}, {'value': 'SubVersionNumber', 'data': win_version['SubVersionNumber']}, {'value': 'VersionNumber', 'data': win_version['VersionNumber']}]}
        if self.config.Arch == Arch.WIN64:
            bundle['HKEY_LOCAL_MACHINE\\Software\\Wow6432Node\\Microsoft\\Windows NT\\CurrentVersion'] = [{'value': 'CSDVersion', 'data': win_version['CSDVersion']}, {'value': 'CurrentBuild', 'data': win_version['CurrentBuild']}, {'value': 'CurrentBuildNumber', 'data': win_version['CurrentBuildNumber']}, {'value': 'CurrentVersion', 'data': win_version['CurrentVersion']}, {'value': 'ProductName', 'data': win_version['ProductName']}, {'value': 'CurrentMinorVersionNumber', 'data': win_version['CurrentMinorVersionNumber'], 'key_type': 'dword'}, {'value': 'CurrentMajorVersionNumber', 'data': win_version['CurrentMajorVersionNumber'], 'key_type': 'dword'}]
        if 'ProductType' in win_version:
            "windows xp 32 doesn't have ProductOptions/ProductType key"
            bundle['HKEY_LOCAL_MACHINE\\System\\CurrentControlSet\\Control\\ProductOptions'] = [{'value': 'ProductType', 'data': win_version['ProductType']}]
        self.reg.import_bundle(bundle)
        wineboot.restart()
        wineboot.update()

    def set_app_default(self, version: str, executable: str):
        if False:
            return 10
        '\n        Change default Windows version per application in a bottle\n        from the given configuration.\n        '
        if version not in win_versions:
            raise ValueError('Given version is not supported.')
        if version == 'winxp' and self.config.Arch == Arch.WIN64:
            version = 'winxp64'
        self.reg.add(key=f'HKEY_CURRENT_USER\\Software\\Wine\\AppDefaults\\{executable}', value='Version', data=version)

    def toggle_virtual_desktop(self, state: bool, resolution: str='800x600'):
        if False:
            for i in range(10):
                print('nop')
        "\n        This function toggles the virtual desktop for a bottle, updating\n        the Desktop's registry key.\n        "
        wineboot = WineBoot(self.config)
        if state:
            self.reg.add(key='HKEY_CURRENT_USER\\Software\\Wine\\Explorer', value='Desktop', data='Default')
            self.reg.add(key='HKEY_CURRENT_USER\\Software\\Wine\\Explorer\\Desktops', value='Default', data=resolution)
        else:
            self.reg.remove(key='HKEY_CURRENT_USER\\Software\\Wine\\Explorer', value='Desktop')
        wineboot.update()

    def apply_cmd_settings(self, scheme=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Change settings for the wine command line in a bottle.\n        This method can also be used to apply the default settings, part\n        of the Bottles experience, these are meant to improve the\n        readability and usability.\n        '
        if scheme is None:
            scheme = {}
        self.reg.import_bundle({'HKEY_CURRENT_USER\\Console\\C:_windows_system32_wineconsole.exe': [{'value': 'ColorTable00', 'data': '2368548'}, {'value': 'CursorSize', 'data': '25'}, {'value': 'CursorVisible', 'data': '1'}, {'value': 'EditionMode', 'data': '0'}, {'value': 'FaceName', 'data': 'Monospace', 'key_type': 'dword'}, {'value': 'FontPitchFamily', 'data': '1'}, {'value': 'FontSize', 'data': '1248584'}, {'value': 'FontWeight', 'data': '400'}, {'value': 'HistoryBufferSize', 'data': '50'}, {'value': 'HistoryNoDup', 'data': '0'}, {'value': 'InsertMode', 'data': '1'}, {'value': 'MenuMask', 'data': '0'}, {'value': 'PopupColors', 'data': '245'}, {'value': 'QuickEdit', 'data': '1'}, {'value': 'ScreenBufferSize', 'data': '9830480'}, {'value': 'ScreenColors', 'data': '11'}, {'value': 'WindowSize', 'data': '1638480'}]})

    def set_renderer(self, value: str):
        if False:
            return 10
        '\n        Set what backend to use for wined3d.\n        '
        if value not in ['gl', 'gdi', 'vulkan']:
            raise ValueError(f'{value} is not a valid renderer (gl, gdi, vulkan)')
        self.reg.add(key='HKEY_CURRENT_USER\\Software\\Wine\\Direct3D', value='renderer', data=value, value_type='REG_SZ')

    def set_dpi(self, value: int):
        if False:
            return 10
        '\n        Set the DPI for a bottle.\n        '
        self.reg.add(key='HKEY_CURRENT_USER\\Control Panel\\Desktop', value='LogPixels', data=str(value), value_type='REG_DWORD')

    def set_grab_fullscreen(self, state: bool):
        if False:
            return 10
        '\n        Set the grab fullscreen setting for a bottle.\n        '
        value = 'Y' if state else 'N'
        self.reg.add(key='HKEY_CURRENT_USER\\Software\\Wine\\X11 Driver', value='GrabFullscreen', data=value)

    def set_take_focus(self, state: bool):
        if False:
            i = 10
            return i + 15
        '\n        Set the take focus setting for a bottle.\n        '
        value = 'Y' if state else 'N'
        self.reg.add(key='HKEY_CURRENT_USER\\Software\\Wine\\X11 Driver', value='UseTakeFocus', data=value)

    def set_decorated(self, state: bool):
        if False:
            return 10
        '\n        Set the decorated setting for a bottle.\n        '
        value = 'Y' if state else 'N'
        self.reg.add(key='HKEY_CURRENT_USER\\Software\\Wine\\X11 Driver', value='Decorated', data=value)

    def set_mouse_warp(self, state: int, executable: str=''):
        if False:
            while True:
                i = 10
        '\n        Set the mouse warp setting for a bottle or a specific executable.\n        Values:\n            0: Disabled\n            1: Enabled\n            2: Forced\n        '
        values = {0: 'disable', 1: 'enable', 2: 'force'}
        if state not in values.keys():
            raise ValueError(f'{state} is not a valid mouse warp setting (0, 1, 2)')
        key = 'HKEY_CURRENT_USER\\Software\\Wine\\DirectInput'
        if executable:
            key = f'HKEY_CURRENT_USER\\Software\\Wine\\AppDefaults\\{executable}\\DirectInput'
        self.reg.add(key=key, value='MouseWarpOverride', data=values[state])