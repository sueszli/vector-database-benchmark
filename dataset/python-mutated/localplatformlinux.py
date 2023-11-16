import os, pwd, grp, sys, logging
from subprocess import call, run, DEVNULL, PIPE, STDOUT
from .customtypes import UserType
logger = logging.getLogger('localplatform')

def _get_user_id() -> int:
    if False:
        return 10
    return pwd.getpwnam(_get_user()).pw_uid

def _get_user() -> str:
    if False:
        for i in range(10):
            print('nop')
    return get_unprivileged_user()

def _get_effective_user_id() -> int:
    if False:
        return 10
    return os.geteuid()

def _get_effective_user() -> str:
    if False:
        print('Hello World!')
    return pwd.getpwuid(_get_effective_user_id()).pw_name

def _get_effective_user_group_id() -> int:
    if False:
        i = 10
        return i + 15
    return os.getegid()

def _get_effective_user_group() -> str:
    if False:
        while True:
            i = 10
    return grp.getgrgid(_get_effective_user_group_id()).gr_name

def _get_user_owner(file_path: str) -> str:
    if False:
        while True:
            i = 10
    return pwd.getpwuid(os.stat(file_path).st_uid).pw_name

def _get_user_group(file_path: str | None=None) -> str:
    if False:
        i = 10
        return i + 15
    return grp.getgrgid(os.stat(file_path).st_gid if file_path is not None else _get_user_group_id()).gr_name

def _get_user_group_id() -> int:
    if False:
        i = 10
        return i + 15
    return pwd.getpwuid(_get_user_id()).pw_gid

def chown(path: str, user: UserType=UserType.HOST_USER, recursive: bool=True) -> bool:
    if False:
        print('Hello World!')
    user_str = ''
    if user == UserType.HOST_USER:
        user_str = _get_user() + ':' + _get_user_group()
    elif user == UserType.EFFECTIVE_USER:
        user_str = _get_effective_user() + ':' + _get_effective_user_group()
    elif user == UserType.ROOT:
        user_str = 'root:root'
    else:
        raise Exception('Unknown User Type')
    result = call(['chown', '-R', user_str, path] if recursive else ['chown', user_str, path])
    return result == 0

def chmod(path: str, permissions: int, recursive: bool=True) -> bool:
    if False:
        for i in range(10):
            print('nop')
    if _get_effective_user_id() != 0:
        return True
    try:
        octal_permissions = int(str(permissions), 8)
        if recursive:
            for (root, dirs, files) in os.walk(path):
                for d in dirs:
                    os.chmod(os.path.join(root, d), octal_permissions)
                for d in files:
                    os.chmod(os.path.join(root, d), octal_permissions)
        os.chmod(path, octal_permissions)
    except:
        return False
    return True

def folder_owner(path: str) -> UserType | None:
    if False:
        for i in range(10):
            print('nop')
    user_owner = _get_user_owner(path)
    if user_owner == _get_user():
        return UserType.HOST_USER
    elif user_owner == _get_effective_user():
        return UserType.EFFECTIVE_USER
    else:
        return None

def get_home_path(user: UserType=UserType.HOST_USER) -> str:
    if False:
        return 10
    user_name = 'root'
    if user == UserType.HOST_USER:
        user_name = _get_user()
    elif user == UserType.EFFECTIVE_USER:
        user_name = _get_effective_user()
    elif user == UserType.ROOT:
        pass
    else:
        raise Exception('Unknown User Type')
    return pwd.getpwnam(user_name).pw_dir

def get_username() -> str:
    if False:
        while True:
            i = 10
    return _get_user()

def setgid(user: UserType=UserType.HOST_USER):
    if False:
        return 10
    user_id = 0
    if user == UserType.HOST_USER:
        user_id = _get_user_group_id()
    elif user == UserType.ROOT:
        pass
    else:
        raise Exception('Unknown user type')
    os.setgid(user_id)

def setuid(user: UserType=UserType.HOST_USER):
    if False:
        print('Hello World!')
    user_id = 0
    if user == UserType.HOST_USER:
        user_id = _get_user_id()
    elif user == UserType.ROOT:
        pass
    else:
        raise Exception('Unknown user type')
    os.setuid(user_id)

async def service_active(service_name: str) -> bool:
    res = run(['systemctl', 'is-active', service_name], stdout=DEVNULL, stderr=DEVNULL)
    return res.returncode == 0

async def service_restart(service_name: str) -> bool:
    call(['systemctl', 'daemon-reload'])
    cmd = ['systemctl', 'restart', service_name]
    res = run(cmd, stdout=PIPE, stderr=STDOUT)
    return res.returncode == 0

async def service_stop(service_name: str) -> bool:
    if not await service_active(service_name):
        return True
    cmd = ['systemctl', 'stop', service_name]
    res = run(cmd, stdout=PIPE, stderr=STDOUT)
    return res.returncode == 0

async def service_start(service_name: str) -> bool:
    if await service_active(service_name):
        return True
    cmd = ['systemctl', 'start', service_name]
    res = run(cmd, stdout=PIPE, stderr=STDOUT)
    return res.returncode == 0

def get_privileged_path() -> str:
    if False:
        for i in range(10):
            print('nop')
    path = os.getenv('PRIVILEGED_PATH')
    if path == None:
        path = get_unprivileged_path()
    return path

def _parent_dir(path: str | None) -> str | None:
    if False:
        i = 10
        return i + 15
    if path == None:
        return None
    if path.endswith('/'):
        path = path[:-1]
    return os.path.dirname(path)

def get_unprivileged_path() -> str:
    if False:
        while True:
            i = 10
    path = os.getenv('UNPRIVILEGED_PATH')
    if path == None:
        path = _parent_dir(os.getenv('PLUGIN_PATH'))
    if path == None:
        logger.debug('Unprivileged path is not properly configured. Making something up!')
        path = _parent_dir(_parent_dir(os.path.realpath(sys.argv[0])))
        if path != None and (not os.path.exists(path)):
            path = None
    if path == None:
        logger.warn('Unprivileged path is not properly configured. Defaulting to /home/deck/homebrew')
        path = '/home/deck/homebrew'
    return path

def get_unprivileged_user() -> str:
    if False:
        for i in range(10):
            print('nop')
    user = os.getenv('UNPRIVILEGED_USER')
    if user == None:
        dir = os.path.realpath(get_unprivileged_path())
        pws = sorted(pwd.getpwall(), reverse=True, key=lambda pw: len(pw.pw_dir))
        for pw in pws:
            if dir.startswith(os.path.realpath(pw.pw_dir)):
                user = pw.pw_name
                break
    if user == None:
        logger.warn("Unprivileged user is not properly configured. Defaulting to 'deck'")
        user = 'deck'
    return user