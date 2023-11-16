from .customtypes import UserType
import os, sys

def chown(path: str, user: UserType=UserType.HOST_USER, recursive: bool=True) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return True

def chmod(path: str, permissions: int, recursive: bool=True) -> bool:
    if False:
        print('Hello World!')
    return True

def folder_owner(path: str) -> UserType | None:
    if False:
        for i in range(10):
            print('nop')
    return UserType.HOST_USER

def get_home_path(user: UserType=UserType.HOST_USER) -> str:
    if False:
        while True:
            i = 10
    return os.path.expanduser('~')

def setgid(user: UserType=UserType.HOST_USER):
    if False:
        i = 10
        return i + 15
    pass

def setuid(user: UserType=UserType.HOST_USER):
    if False:
        while True:
            i = 10
    pass

async def service_active(service_name: str) -> bool:
    return True

async def service_stop(service_name: str) -> bool:
    return True

async def service_start(service_name: str) -> bool:
    return True

async def service_restart(service_name: str) -> bool:
    if service_name == 'plugin_loader':
        sys.exit(42)
    return True

def get_username() -> str:
    if False:
        print('Hello World!')
    return os.getlogin()

def get_privileged_path() -> str:
    if False:
        return 10
    'On windows, privileged_path is equal to unprivileged_path'
    return get_unprivileged_path()

def get_unprivileged_path() -> str:
    if False:
        return 10
    path = os.getenv('UNPRIVILEGED_PATH')
    if path == None:
        path = os.getenv('PRIVILEGED_PATH', os.path.join(os.path.expanduser('~'), 'homebrew'))
    return path

def get_unprivileged_user() -> str:
    if False:
        return 10
    return os.getenv('UNPRIVILEGED_USER', os.getlogin())