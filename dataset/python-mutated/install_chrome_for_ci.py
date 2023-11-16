"""Install Chrome for CI server.

This is meant to be run as part of CI setup. It installs the correct
version of Chrome for our CI tests and checks that the correct version
was installed.
"""
from __future__ import annotations
import re
from scripts import common
CHROME_VERSION = '102.0.5005.61-1'
URL_TEMPLATE = 'https://github.com/webnicer/chrome-downloads/raw/master/x64.deb/google-chrome-stable_{}_amd64.deb'
CHROME_DEB_FILE = 'google-chrome.deb'

def install_chrome(version: str) -> None:
    if False:
        return 10
    'Install Chrome from the URL in URL_TEMPLATE.\n\n    Args:\n        version: str. The version of Chrome to install. This must be one\n            of the versions available from\n            github.com/webnicer/chrome-downloads.\n    '
    common.run_cmd(['sudo', 'apt-get', 'update'])
    common.run_cmd(['sudo', 'apt-get', 'install', 'libappindicator3-1'])
    common.run_cmd(['curl', '-L', '-o', CHROME_DEB_FILE, URL_TEMPLATE.format(version)])
    common.run_cmd(['sudo', 'sed', '-i', 's|HERE/chrome\\"|HERE/chrome\\" --disable-setuid-sandbox|g', '/opt/google/chrome/google-chrome'])
    common.run_cmd(['sudo', 'dpkg', '-i', CHROME_DEB_FILE])

def get_chrome_version() -> str:
    if False:
        for i in range(10):
            print('nop')
    'Get the current version of Chrome.\n\n    Note that this only works on Linux systems. On macOS, for example,\n    the `google-chrome` command may not work.\n\n    Returns:\n        str. The version of Chrome we found.\n    '
    output = str(common.run_cmd(['google-chrome', '--version']))
    chrome_version = ''.join(re.findall('([0-9]|\\.)', output))
    return chrome_version

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Install Chrome and check the correct version was installed.'
    install_chrome(CHROME_VERSION)
    found_version = get_chrome_version()
    if not CHROME_VERSION.startswith(found_version):
        raise RuntimeError('Chrome version {} should have been installed. Version {} was found instead.'.format(CHROME_VERSION, found_version))
    print('Chrome version {} installed.'.format(found_version))
if __name__ == '__main__':
    main()