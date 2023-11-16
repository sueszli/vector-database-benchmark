from persepolis.scripts.useful_tools import determineConfigFolder
from persepolis.scripts import osCommands
from persepolis.constants import OS, BROWSER
import subprocess
import platform
import sys
import os
os_type = platform.system()
home_address = str(os.path.expanduser('~'))
config_folder = determineConfigFolder()

def browserIntegration(browser):
    if False:
        return 10
    if os_type == OS.LINUX:
        exec_path = os.path.join(config_folder, 'persepolis_run_shell')
        if browser == BROWSER.CHROMIUM:
            native_message_folder = home_address + '/.config/chromium/NativeMessagingHosts'
        elif browser == BROWSER.CHROME:
            native_message_folder = home_address + '/.config/google-chrome/NativeMessagingHosts'
        elif browser == BROWSER.FIREFOX:
            native_message_folder = home_address + '/.mozilla/native-messaging-hosts'
        elif browser == BROWSER.VIVALDI:
            native_message_folder = home_address + '/.config/vivaldi/NativeMessagingHosts'
        elif browser == BROWSER.OPERA:
            native_message_folder = home_address + '/.config/opera/NativeMessagingHosts'
        elif browser == BROWSER.BRAVE:
            native_message_folder = home_address + '/.config/BraveSoftware/Brave-Browser/NativeMessagingHosts'
    elif os_type in OS.BSD_FAMILY:
        exec_path = os.path.join(config_folder, 'persepolis_run_shell')
        if browser == BROWSER.CHROMIUM:
            native_message_folder = home_address + '/.config/chromium/NativeMessagingHosts'
        elif browser == BROWSER.CHROME:
            native_message_folder = home_address + '/.config/google-chrome/NativeMessagingHosts'
        elif browser == BROWSER.FIREFOX:
            native_message_folder = home_address + '/.mozilla/native-messaging-hosts'
        elif browser == BROWSER.VIVALDI:
            native_message_folder = home_address + '/.config/vivaldi/NativeMessagingHosts'
        elif browser == BROWSER.OPERA:
            native_message_folder = home_address + '/.config/opera/NativeMessagingHosts'
        elif browser == BROWSER.BRAVE:
            native_message_folder = home_address + '/.config/BraveSoftware/Brave-Browser/NativeMessagingHosts'
    elif os_type == OS.OSX:
        exec_path = os.path.join(config_folder, 'persepolis_run_shell')
        if browser == BROWSER.CHROMIUM:
            native_message_folder = home_address + '/Library/Application Support/Chromium/NativeMessagingHosts'
        elif browser == BROWSER.CHROME:
            native_message_folder = home_address + '/Library/Application Support/Google/Chrome/NativeMessagingHosts'
        elif browser == BROWSER.FIREFOX:
            native_message_folder = home_address + '/Library/Application Support/Mozilla/NativeMessagingHosts'
        elif browser == BROWSER.VIVALDI:
            native_message_folder = home_address + '/Library/Application Support/Vivaldi/NativeMessagingHosts'
        elif browser == BROWSER.OPERA:
            native_message_folder = home_address + '/Library/Application Support/Opera/NativeMessagingHosts/'
        elif browser == BROWSER.BRAVE:
            native_message_folder = home_address + '/Library/Application Support/BraveSoftware/Brave-Browser/NativeMessagingHosts/'
    elif os_type == OS.WINDOWS:
        cwd = sys.argv[0]
        current_directory = os.path.dirname(cwd)
        exec_path = os.path.join(current_directory, 'Persepolis Download Manager.exe')
        exec_path = exec_path.replace('\\', '\\\\')
        if browser in BROWSER.CHROME_FAMILY:
            native_message_folder = os.path.join(home_address, 'AppData\\Local\\persepolis_download_manager', 'chrome')
        else:
            native_message_folder = os.path.join(home_address, 'AppData\\Local\\persepolis_download_manager', 'firefox')
    webextension_json_connector = {'name': 'com.persepolis.pdmchromewrapper', 'type': 'stdio', 'path': str(exec_path), 'description': 'Integrate Persepolis with %s using WebExtensions' % browser}
    if browser in BROWSER.CHROME_FAMILY:
        webextension_json_connector['allowed_origins'] = ['chrome-extension://legimlagjjoghkoedakdjhocbeomojao/']
    elif browser == BROWSER.FIREFOX:
        webextension_json_connector['allowed_extensions'] = ['com.persepolis.pdmchromewrapper@persepolisdm.github.io', 'com.persepolis.pdmchromewrapper.offline@persepolisdm.github.io']
    native_message_file = os.path.join(native_message_folder, 'com.persepolis.pdmchromewrapper.json')
    osCommands.makeDirs(native_message_folder)
    f = open(native_message_file, 'w')
    f.write(str(webextension_json_connector).replace("'", '"'))
    f.close()
    if os_type != OS.WINDOWS:
        pipe_json = subprocess.Popen(['chmod', '+x', str(native_message_file)], stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)
        if pipe_json.wait() == 0:
            json_done = True
        else:
            json_done = False
    else:
        native_done = None
        import winreg
        if browser in BROWSER.CHROME_FAMILY:
            try:
                winreg.CreateKey(winreg.HKEY_CURRENT_USER, 'SOFTWARE\\Google\\Chrome\\NativeMessagingHosts\\com.persepolis.pdmchromewrapper')
                gintKey = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'SOFTWARE\\Google\\Chrome\\NativeMessagingHosts\\com.persepolis.pdmchromewrapper', 0, winreg.KEY_ALL_ACCESS)
                winreg.SetValueEx(gintKey, '', 0, winreg.REG_SZ, native_message_file)
                winreg.CloseKey(gintKey)
                json_done = True
            except WindowsError:
                json_done = False
        elif browser == BROWSER.FIREFOX:
            try:
                winreg.CreateKey(winreg.HKEY_CURRENT_USER, 'SOFTWARE\\Mozilla\\NativeMessagingHosts\\com.persepolis.pdmchromewrapper')
                fintKey = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'SOFTWARE\\Mozilla\\NativeMessagingHosts\\com.persepolis.pdmchromewrapper', 0, winreg.KEY_ALL_ACCESS)
                winreg.SetValueEx(fintKey, '', 0, winreg.REG_SZ, native_message_file)
                winreg.CloseKey(fintKey)
                json_done = True
            except WindowsError:
                json_done = False
    if os_type in OS.UNIX_LIKE + [OS.OSX]:
        shell_list = ['/bin/bash', '/usr/local/bin/bash', '/bin/sh', '/usr/local/bin/sh', '/bin/ksh', '/bin/tcsh']
        for shell in shell_list:
            if os.path.isfile(shell):
                shebang = '#!' + shell
                break
        if os_type == OS.OSX:
            cwd = sys.argv[0]
            current_directory = os.path.dirname(cwd)
            persepolis_path = os.path.join(current_directory, 'Persepolis Download Manager')
        else:
            persepolis_path = 'persepolis'
        persepolis_run_shell_contents = shebang + '\n' + '"' + persepolis_path + '" "$@"'
        f = open(exec_path, 'w')
        f.writelines(persepolis_run_shell_contents)
        f.close()
        pipe_native = subprocess.Popen(['chmod', '+x', exec_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)
        if pipe_native.wait() == 0:
            native_done = True
        else:
            native_done = False
    return (json_done, native_done)