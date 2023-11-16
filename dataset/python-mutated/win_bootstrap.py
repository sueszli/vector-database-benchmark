import subprocess
from typing import Callable
from typing import List

class Requirement:

    def __init__(self, full_name: str, choco_name: str, detect: Callable, extras: str='') -> None:
        if False:
            while True:
                i = 10
        self.full_name = full_name
        self.choco_name = choco_name
        self.detect = detect
        self.extras = extras

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.full_name
install_choco_pwsh = "\n[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072;\nInvoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'));\n"
install_wsl2_pwsh = '\nwsl --update; wsl --shutdown; wsl --set-default-version 2; wsl --install -d Ubuntu; wsl --setdefault Ubuntu;\n'

def make_admin_cmd(admin_cmd: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return f'Start-Process PowerShell -Wait -Verb RunAs -ArgumentList "Set-ExecutionPolicy Bypass -Scope Process -Force; {admin_cmd}; "'

def where_is(binary: str, req: Requirement) -> bool:
    if False:
        print('Hello World!')
    print(f'{req.full_name} - {binary}', end='', flush=True)
    found = path_where_is(binary)
    if not found:
        found = full_where_is(binary)
    if found:
        print(' √')
    else:
        print(' ×')
    return found

def path_where_is(binary: str) -> bool:
    if False:
        print('Hello World!')
    try:
        cmds = ['where.exe', binary]
        output = subprocess.run(cmds, capture_output=True, cwd='C:\\')
        out = str(output.stdout.decode('utf-8')).split('\r\n')
        if binary in out[0]:
            return True
    except Exception as e:
        print('error', e)
        pass
    return False

def full_where_is(binary: str) -> bool:
    if False:
        while True:
            i = 10
    try:
        powershell_cmd = f'where.exe /R C:\\ *.exe | findstr \\{binary}$'
        cmds = ['powershell.exe', '-Command', powershell_cmd]
        output = subprocess.run(cmds, capture_output=True, cwd='C:\\')
        out = str(output.stdout.decode('utf-8')).split('\r\n')
        if binary in out[0]:
            return True
    except Exception as e:
        print('error', e)
        pass
    return False

def exe(binary: str) -> Callable:
    if False:
        for i in range(10):
            print('nop')

    def call(req: Requirement) -> bool:
        if False:
            return 10
        return where_is(binary=binary, req=req)
    return call

def detect_wsl2(req: Requirement) -> bool:
    if False:
        print('Hello World!')
    print(f'{req.full_name} - wsl.exe ', end='')
    try:
        powershell_cmd = 'wsl.exe --status'
        cmds = ['powershell.exe', '-Command', powershell_cmd]
        output = subprocess.run(cmds, capture_output=True)
        out = output.stdout.decode('utf-16')
        if 'Default Distribution: Ubuntu' in out:
            pass
        if 'Default Version: 2' in out:
            print(' √')
            return True
    except Exception as e:
        print('error', e)
        pass
    print(' ×')
    return False
requirements = []
requirements.append(Requirement(full_name='Windows Subsystem for Linux 2', choco_name='wsl2', detect=detect_wsl2))
requirements.append(Requirement(full_name='Chocolatey Package Manager', choco_name='choco', detect=exe('choco.exe')))
requirements.append(Requirement(full_name='Anaconda Individual Edition', choco_name='anaconda3', detect=exe('conda.exe')))
requirements.append(Requirement(full_name='Git Version Control', choco_name='git', detect=exe('git.exe')))
requirements.append(Requirement(full_name='Docker Desktop', choco_name='docker-desktop', detect=exe('docker.exe')))

def install_elevated_powershell(full_name: str, powershell_cmd: str) -> None:
    if False:
        while True:
            i = 10
    try:
        input(f'\nInstalling {full_name} requires Administrator.\nWhen the UAC dialogue appears click Yes on the left.\n\nPress enter to start...')
        powershell_cmds = ['-command', powershell_cmd]
        output = subprocess.run(['powershell.exe'] + powershell_cmds, capture_output=True)
        _ = output.stdout.decode('utf-8')
    except Exception as e:
        print('failed', e)

def install_choco() -> None:
    if False:
        i = 10
        return i + 15
    return install_elevated_powershell(full_name='Chocolatey', powershell_cmd=make_admin_cmd(install_choco_pwsh))

def install_wsl2() -> None:
    if False:
        print('Hello World!')
    return install_elevated_powershell(full_name='WSL2', powershell_cmd=make_admin_cmd(install_wsl2_pwsh))

def install_deps(requirements: List[Requirement]) -> None:
    if False:
        return 10
    package_names = []
    for req in requirements:
        package_names.append(req.choco_name)
    try:
        input('\nInstalling packages requires Administrator.\nWhen the UAC dialogue appears click Yes on the left.\n\nPress enter to start...')
        choco_args = f"choco.exe install {' '.join(package_names)} -y"
        powershell_cmds = ['-command', make_admin_cmd(choco_args)]
        output = subprocess.run(['powershell.exe'] + powershell_cmds, capture_output=True)
        _ = str(output.stdout.decode('utf-8'))
    except Exception as e:
        print('failed', e)

def ask_install(requirement: Requirement) -> bool:
    if False:
        while True:
            i = 10
    val = input(f'Do you want to install {requirement.full_name} (Y/n): ')
    if 'y' in val.lower():
        return True
    return False

def check_all(requirements: List[Requirement]) -> List[Requirement]:
    if False:
        print('Hello World!')
    missing = []
    for req in requirements:
        if not req.detect(req):
            missing.append(req)
    return missing

def main() -> None:
    if False:
        while True:
            i = 10
    print('\nHAGrid Windows Dependency Installer')
    print('===================================\n')
    print('Searching your computer for:')
    missing_deps = check_all(requirements=requirements)
    if len(missing_deps) > 0:
        print('\nWe were unable to find the following dependencies:')
        print('-----------------------------------')
        for dep in missing_deps:
            print(f'{dep.full_name}')
    print('')
    desired = []
    choco_required = False
    wsl2_required = False
    for dep in missing_deps:
        if ask_install(dep):
            if dep.choco_name == 'choco':
                choco_required = True
            elif dep.choco_name == 'wsl2':
                wsl2_required = True
            else:
                desired.append(dep)
        elif dep.choco_name == 'choco':
            print('You must install Chocolatey to install other dependencies')
            return
    if wsl2_required:
        install_wsl2()
    if choco_required:
        install_choco()
    if len(desired) > 0:
        install_deps(desired)
    print('')
    still_missing = check_all(requirements=missing_deps)
    if len(still_missing) > 0:
        print('We were still unable to find the following dependencies:')
        print('-----------------------------------')
        for dep in still_missing:
            print(f'{dep.full_name}')
        print('Please try again.')
    else:
        print('\nCongratulations. All done.')
        print('===================================\n')
        print('Now you can run HAGrid on Windows!')
if __name__ == '__main__':
    main()