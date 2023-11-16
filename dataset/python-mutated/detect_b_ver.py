"""Detect the browser version before launching tests.
Eg. detect_b_ver.get_browser_version_from_os("google-chrome")"""
import datetime
import os
import platform
import re
import subprocess
import sys

class File(object):

    def __init__(self, stream):
        if False:
            while True:
                i = 10
        self.content = stream.content
        self.__stream = stream
        self.__temp_name = 'driver'

    @property
    def filename(self):
        if False:
            return 10
        try:
            filename = re.findall('filename=(.+)', self.__stream.headers['content-disposition'])[0]
        except KeyError:
            filename = '%s.zip' % self.__temp_name
        except IndexError:
            filename = '%s.exe' % self.__temp_name
        if '"' in filename:
            filename = filename.replace('"', '')
        return filename

class OSType(object):
    LINUX = 'linux'
    MAC = 'mac'
    WIN = 'win'

class ChromeType(object):
    GOOGLE = 'google-chrome'
    MSEDGE = 'edge'
PATTERN = {ChromeType.GOOGLE: '\\d+\\.\\d+\\.\\d+', ChromeType.MSEDGE: '\\d+\\.\\d+\\.\\d+'}

def os_name():
    if False:
        return 10
    if 'linux' in sys.platform:
        return OSType.LINUX
    elif 'darwin' in sys.platform:
        return OSType.MAC
    elif 'win32' in sys.platform:
        return OSType.WIN
    else:
        raise Exception('Could not determine the OS type!')

def os_architecture():
    if False:
        print('Hello World!')
    if platform.machine().endswith('64'):
        return 64
    else:
        return 32

def os_type():
    if False:
        i = 10
        return i + 15
    return '%s%s' % (os_name(), os_architecture())

def is_arch(os_sys_type):
    if False:
        for i in range(10):
            print('nop')
    if '_m1' in os_sys_type:
        return True
    return platform.processor() != 'i386'

def is_mac_os(os_sys_type):
    if False:
        while True:
            i = 10
    return OSType.MAC in os_sys_type

def get_date_diff(date1, date2, date_format):
    if False:
        print('Hello World!')
    a = datetime.datetime.strptime(date1, date_format)
    b = datetime.datetime.strptime(str(date2.strftime(date_format)), date_format)
    return (b - a).days

def linux_browser_apps_to_cmd(*apps):
    if False:
        print('Hello World!')
    "Create 'browser --version' command from browser app names."
    ignore_errors_cmd_part = ' 2>/dev/null' if os.getenv('WDM_LOG_LEVEL') == '0' else ''
    return ' || '.join(('%s --version%s' % (i, ignore_errors_cmd_part) for i in apps))

def chrome_on_linux_path(prefer_chromium=False):
    if False:
        i = 10
        return i + 15
    if os_name() != 'linux':
        return ''
    if prefer_chromium:
        paths = ['/bin/chromium', '/bin/chromium-browser']
        for path in paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
    paths = ['/bin/google-chrome', '/bin/google-chrome-stable']
    for path in paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    paths = os.environ['PATH'].split(os.pathsep)
    binaries = []
    binaries.append('google-chrome')
    binaries.append('google-chrome-stable')
    binaries.append('chrome')
    binaries.append('chromium')
    binaries.append('chromium-browser')
    binaries.append('google-chrome-beta')
    binaries.append('google-chrome-dev')
    binaries.append('google-chrome-unstable')
    for binary in binaries:
        for path in paths:
            full_path = os.path.join(path, binary)
            if os.path.exists(full_path) and os.access(full_path, os.X_OK):
                return full_path
    return '/usr/bin/google-chrome'

def edge_on_linux_path():
    if False:
        return 10
    if os_name() != 'linux':
        return ''
    paths = os.environ['PATH'].split(os.pathsep)
    binaries = []
    binaries.append('microsoft-edge')
    binaries.append('microsoft-edge-stable')
    binaries.append('microsoft-edge-beta')
    binaries.append('microsoft-edge-dev')
    for binary in binaries:
        for path in paths:
            full_path = os.path.join(path, binary)
            if os.path.exists(full_path) and os.access(full_path, os.X_OK):
                return full_path
    return '/usr/bin/microsoft-edge'

def chrome_on_windows_path():
    if False:
        return 10
    if os_name() != 'win32':
        return ''
    candidates = []
    for item in map(os.environ.get, ('PROGRAMFILES', 'PROGRAMFILES(X86)', 'LOCALAPPDATA', 'PROGRAMW6432')):
        for subitem in ('Google/Chrome/Application', 'Google/Chrome Beta/Application', 'Google/Chrome Canary/Application'):
            try:
                candidates.append(os.sep.join((item, subitem, 'chrome.exe')))
            except TypeError:
                pass
    for candidate in candidates:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return os.path.normpath(candidate)
    return ''

def edge_on_windows_path():
    if False:
        return 10
    if os_name() != 'win32':
        return ''
    candidates = []
    for item in map(os.environ.get, ('PROGRAMFILES', 'PROGRAMFILES(X86)', 'LOCALAPPDATA', 'PROGRAMW6432')):
        for subitem in ('Microsoft/Edge/Application', 'Microsoft/Edge Beta/Application', 'Microsoft/Edge Canary/Application'):
            try:
                candidates.append(os.sep.join((item, subitem, 'msedge.exe')))
            except TypeError:
                pass
    for candidate in candidates:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return os.path.normpath(candidate)
    return ''

def windows_browser_apps_to_cmd(*apps):
    if False:
        i = 10
        return i + 15
    'Create analogue of browser --version command for windows.'
    powershell = determine_powershell()
    first_hit_template = '$tmp = {expression}; if ($tmp) {{echo $tmp; Exit;}};'
    script = "$ErrorActionPreference='silentlycontinue'; " + ' '.join((first_hit_template.format(expression=e) for e in apps))
    return '%s -NoProfile "%s"' % (powershell, script)

def get_binary_location(browser_type, prefer_chromium=False):
    if False:
        print('Hello World!')
    'Return the full path of the browser binary.\n    If going for better results in UC Mode, use: prefer_chromium=True'
    cmd_mapping = {ChromeType.GOOGLE: {OSType.LINUX: chrome_on_linux_path(prefer_chromium), OSType.MAC: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', OSType.WIN: chrome_on_windows_path()}, ChromeType.MSEDGE: {OSType.LINUX: edge_on_linux_path(), OSType.MAC: '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge', OSType.WIN: edge_on_windows_path()}}
    return cmd_mapping[browser_type][os_name()]

def get_browser_version_from_binary(binary_location):
    if False:
        print('Hello World!')
    try:
        if binary_location.count('\\ ') != binary_location.count(' '):
            binary_location = binary_location.replace(' ', '\\ ')
        cmd_mapping = binary_location + ' --version'
        pattern = '\\d+\\.\\d+\\.\\d+'
        quad_pattern = '\\d+\\.\\d+\\.\\d+\\.\\d+'
        quad_version = read_version_from_cmd(cmd_mapping, quad_pattern)
        if quad_version and len(str(quad_version)) >= 9:
            return quad_version
        version = read_version_from_cmd(cmd_mapping, pattern)
        return version
    except Exception:
        return None

def get_browser_version_from_os(browser_type):
    if False:
        return 10
    'Return installed browser version.'
    cmd_mapping = {ChromeType.GOOGLE: {OSType.LINUX: linux_browser_apps_to_cmd('google-chrome', 'google-chrome-stable', 'chrome', 'chromium', 'chromium-browser', 'google-chrome-beta', 'google-chrome-dev', 'google-chrome-unstable'), OSType.MAC: '/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --version', OSType.WIN: windows_browser_apps_to_cmd('(Get-Item -Path "$env:PROGRAMFILES\\Google\\Chrome\\Application\\chrome.exe").VersionInfo.FileVersion', '(Get-Item -Path "$env:PROGRAMFILES (x86)\\Google\\Chrome\\Application\\chrome.exe").VersionInfo.FileVersion', '(Get-Item -Path "$env:LOCALAPPDATA\\Google\\Chrome\\Application\\chrome.exe").VersionInfo.FileVersion', '(Get-ItemProperty -Path Registry::"HKCU\\SOFTWARE\\Google\\Chrome\\BLBeacon").version', '(Get-ItemProperty -Path Registry::"HKLM\\SOFTWARE\\Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Google Chrome").version')}, ChromeType.MSEDGE: {OSType.LINUX: linux_browser_apps_to_cmd('microsoft-edge', 'microsoft-edge-stable', 'microsoft-edge-beta', 'microsoft-edge-dev'), OSType.MAC: '/Applications/Microsoft\\ Edge.app/Contents/MacOS/Microsoft\\ Edge --version', OSType.WIN: windows_browser_apps_to_cmd('(Get-Item -Path "$env:PROGRAMFILES\\Microsoft\\Edge\\Application\\msedge.exe").VersionInfo.FileVersion', '(Get-Item -Path "$env:PROGRAMFILES (x86)\\Microsoft\\Edge\\Application\\msedge.exe").VersionInfo.FileVersion', '(Get-ItemProperty -Path Registry::"HKCU\\SOFTWARE\\Microsoft\\Edge\\BLBeacon").version', '(Get-ItemProperty -Path Registry::"HKLM\\SOFTWARE\\Microsoft\\EdgeUpdate\\Clients\\{56EB18F8-8008-4CBD-B6D2-8C97FE7E9062}").pv', '(Get-Item -Path "$env:LOCALAPPDATA\\Microsoft\\Edge Beta\\Application\\msedge.exe").VersionInfo.FileVersion', '(Get-Item -Path "$env:PROGRAMFILES\\Microsoft\\Edge Beta\\Application\\msedge.exe").VersionInfo.FileVersion', '(Get-Item -Path "$env:PROGRAMFILES (x86)\\Microsoft\\Edge Beta\\Application\\msedge.exe").VersionInfo.FileVersion', '(Get-ItemProperty -Path Registry::"HKCU\\SOFTWARE\\Microsoft\\Edge Beta\\BLBeacon").version', '(Get-Item -Path "$env:LOCALAPPDATA\\Microsoft\\Edge Dev\\Application\\msedge.exe").VersionInfo.FileVersion', '(Get-Item -Path "$env:PROGRAMFILES\\Microsoft\\Edge Dev\\Application\\msedge.exe").VersionInfo.FileVersion', '(Get-Item -Path "$env:PROGRAMFILES (x86)\\Microsoft\\Edge Dev\\Application\\msedge.exe").VersionInfo.FileVersion', '(Get-ItemProperty -Path Registry::"HKCU\\SOFTWARE\\Microsoft\\Edge Dev\\BLBeacon").version', '(Get-Item -Path "$env:LOCALAPPDATA\\Microsoft\\Edge SxS\\Application\\msedge.exe").VersionInfo.FileVersion', '(Get-ItemProperty -Path Registry::"HKCU\\SOFTWARE\\Microsoft\\Edge SxS\\BLBeacon").version', "(Get-Item (Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\msedge.exe').'(Default)').VersionInfo.ProductVersion", "[System.Diagnostics.FileVersionInfo]::GetVersionInfo((Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\msedge.exe').'(Default)').ProductVersion", 'Get-AppxPackage -Name *MicrosoftEdge.* | Foreach Version', '(Get-ItemProperty -Path Registry::"HKLM\\SOFTWARE\\Wow6432Node\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Microsoft Edge").version')}}
    try:
        cmd_mapping = cmd_mapping[browser_type][os_name()]
        pattern = PATTERN[browser_type]
        quad_pattern = '\\d+\\.\\d+\\.\\d+\\.\\d+'
        quad_version = read_version_from_cmd(cmd_mapping, quad_pattern)
        if quad_version and len(str(quad_version)) >= 9:
            return quad_version
        version = read_version_from_cmd(cmd_mapping, pattern)
        return version
    except Exception:
        raise Exception('Can not find browser %s installed in your system!' % browser_type)

def format_version(browser_type, version):
    if False:
        while True:
            i = 10
    if not version or version == 'latest':
        return 'latest'
    try:
        pattern = PATTERN[browser_type]
        result = re.search(pattern, version)
        return result.group(0) if result else version
    except Exception:
        return 'latest'

def get_browser_version(browser_type, metadata):
    if False:
        return 10
    pattern = PATTERN[browser_type]
    version_from_os = metadata['version']
    result = re.search(pattern, version_from_os)
    version = result.group(0) if version_from_os else None
    return version

def read_version_from_cmd(cmd, pattern):
    if False:
        for i in range(10):
            print('nop')
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, shell=True) as stream:
        stdout = stream.communicate()[0].decode()
        version = re.search(pattern, stdout)
        version = version.group(0) if version else None
    return version

def determine_powershell():
    if False:
        while True:
            i = 10
    'Returns "True" if runs in Powershell and "False" if another console.'
    cmd = '(dir 2>&1 *`|echo CMD);&<# rem #>echo powershell'
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, shell=True) as stream:
        stdout = stream.communicate()[0].decode()
    return '' if stdout == 'powershell' else 'powershell'