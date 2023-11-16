from persepolis.constants import OS
import subprocess
import platform
import shutil
import os
os_type = platform.system()
home_address = os.path.expanduser('~')

def findFileManager():
    if False:
        while True:
            i = 10
    pipe = subprocess.check_output(['xdg-mime', 'query', 'default', 'inode/directory'])
    file_manager = pipe.decode('utf-8').strip().lower()
    return file_manager

def touch(file_path):
    if False:
        print('Hello World!')
    if not os.path.isfile(file_path):
        f = open(file_path, 'w')
        f.close()

def xdgOpen(file_path, f_type='file', path='file'):
    if False:
        while True:
            i = 10
    if f_type == 'folder' and path == 'file':
        highlight = True
    else:
        highlight = False
    if os_type in OS.UNIX_LIKE:
        file_manager = findFileManager()
        if highlight:
            if 'dolphin' in file_manager:
                subprocess.Popen(['dolphin', '--select', file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)
            elif 'dde-file-manager' in file_manager:
                subprocess.Popen(['dde-file-manager', '--show-item', file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)
            elif file_manager in ['org.gnome.nautilus.desktop', 'nemo.desktop', 'io.elementary.files.desktop']:
                if 'nautilus' in file_manager:
                    file_manager = 'nautilus'
                elif 'elementary' in file_manager:
                    file_manager = 'io.elementary.files'
                elif 'nemo' in file_manager:
                    file_manager = 'nemo'
                subprocess.Popen([file_manager, file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)
            else:
                file_name = os.path.basename(str(file_path))
                file_path_split = file_path.split(file_name)
                del file_path_split[-1]
                folder_path = file_name.join(file_path_split)
                subprocess.Popen(['xdg-open', folder_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)
        else:
            subprocess.Popen(['xdg-open', file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)
    elif os_type == OS.OSX:
        if highlight:
            subprocess.Popen(['open', '-R', file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)
        else:
            subprocess.Popen(['open', file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False)
    elif os_type == OS.WINDOWS:
        CREATE_NO_WINDOW = 134217728
        if highlight:
            subprocess.Popen(['explorer.exe', '/select,', file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False, creationflags=CREATE_NO_WINDOW)
        else:
            subprocess.Popen(['cmd', '/C', 'start', file_path, file_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False, creationflags=CREATE_NO_WINDOW)

def remove(file_path):
    if False:
        print('Hello World!')
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
            return 'ok'
        except:
            return 'cant'
    else:
        return 'no'

def removeDir(folder_path):
    if False:
        print('Hello World!')
    if os.path.isdir(folder_path):
        try:
            shutil.rmtree(folder_path)
            return 'ok'
        except:
            return 'cant'
    else:
        return 'no'

def makeDirs(folder_path, hidden=False):
    if False:
        for i in range(10):
            print('nop')
    if hidden:
        if os_type == OS.WINDOWS:
            os.makedirs(folder_path, exist_ok=True)
            CREATE_NO_WINDOW = 134217728
            subprocess.Popen(['attrib', '+h', folder_path], stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=False, creationflags=CREATE_NO_WINDOW)
        else:
            dir_name = os.path.basename(folder_path)
            dir_name = '.' + dir_name
            folder_path = os.path.join(os.path.dirname(folder_path), dir_name)
            os.makedirs(folder_path, exist_ok=True)
    else:
        os.makedirs(folder_path, exist_ok=True)
    return folder_path

def findMountPoint(path):
    if False:
        for i in range(10):
            print('nop')
    while not os.path.ismount(path):
        path = os.path.dirname(path)
    return path

def makeTempDownloadDir(path):
    if False:
        print('Hello World!')
    if os.lstat(path).st_dev == os.lstat(home_address):
        if os_type != OS.WINDOWS:
            download_path_temp = os.path.join(home_address, '.persepolis')
        else:
            download_path_temp = os.path.join(home_address, 'AppData', 'Local', 'persepolis')
        download_path_temp = makeDirs(download_path_temp)
    else:
        mount_point = findMountPoint(path)
        download_path_temp = os.path.join(mount_point, 'persepolis')
        download_path_temp = makeDirs(download_path_temp, hidden=True)
    return download_path_temp

def moveFile(old_file_path, new_path, new_path_type='folder'):
    if False:
        while True:
            i = 10
    if os.path.isfile(old_file_path):
        if new_path_type == 'folder':
            check_path = os.path.isdir(new_path)
        else:
            check_path = True
        if check_path:
            try:
                shutil.move(old_file_path, new_path)
                return True
            except:
                return False
        else:
            return False
    else:
        return False