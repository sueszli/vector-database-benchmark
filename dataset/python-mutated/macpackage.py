"""
Install pkg, dmg and .app applications on macOS minions.

"""
import logging
import os
import shlex
import salt.utils.platform
log = logging.getLogger(__name__)
__virtualname__ = 'macpackage'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only work on Mac OS\n    '
    if salt.utils.platform.is_darwin():
        return __virtualname__
    return (False, 'Only available on Mac OS systems with pipes')

def install(pkg, target='LocalSystem', store=False, allow_untrusted=False):
    if False:
        while True:
            i = 10
    "\n    Install a pkg file\n\n    Args:\n        pkg (str): The package to install\n        target (str): The target in which to install the package to\n        store (bool): Should the package be installed as if it was from the\n                      store?\n        allow_untrusted (bool): Allow the installation of untrusted packages?\n\n    Returns:\n        dict: A dictionary containing the results of the installation\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' macpackage.install test.pkg\n    "
    if '*.' not in pkg:
        pkg = shlex.quote(pkg)
    target = shlex.quote(target)
    cmd = f'installer -pkg {pkg} -target {target}'
    if store:
        cmd += ' -store'
    if allow_untrusted:
        cmd += ' -allowUntrusted'
    python_shell = False
    if '*.' in cmd:
        python_shell = True
    return __salt__['cmd.run_all'](cmd, python_shell=python_shell)

def install_app(app, target='/Applications/'):
    if False:
        while True:
            i = 10
    "\n    Install an app file by moving it into the specified Applications directory\n\n    Args:\n        app (str): The location of the .app file\n        target (str): The target in which to install the package to\n                      Default is ''/Applications/''\n\n    Returns:\n        str: The results of the rsync command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' macpackage.install_app /tmp/tmp.app /Applications/\n    "
    if target[-4:] != '.app':
        if app[-1:] == '/':
            base_app = os.path.basename(app[:-1])
        else:
            base_app = os.path.basename(app)
        target = os.path.join(target, base_app)
    if not app[-1] == '/':
        app += '/'
    cmd = f'rsync -a --delete "{app}" "{target}"'
    return __salt__['cmd.run'](cmd)

def uninstall_app(app):
    if False:
        i = 10
        return i + 15
    "\n    Uninstall an app file by removing it from the Applications directory\n\n    Args:\n        app (str): The location of the .app file\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' macpackage.uninstall_app /Applications/app.app\n    "
    return __salt__['file.remove'](app)

def mount(dmg):
    if False:
        return 10
    "\n    Attempt to mount a dmg file to a temporary location and return the\n    location of the pkg file inside\n\n    Args:\n        dmg (str): The location of the dmg file to mount\n\n    Returns:\n        tuple: Tuple containing the results of the command along with the mount\n               point\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' macpackage.mount /tmp/software.dmg\n    "
    temp_dir = __salt__['temp.dir'](prefix='dmg-')
    cmd = f'hdiutil attach -readonly -nobrowse -mountpoint {temp_dir} "{dmg}"'
    return (__salt__['cmd.run'](cmd), temp_dir)

def unmount(mountpoint):
    if False:
        print('Hello World!')
    "\n    Attempt to unmount a dmg file from a temporary location\n\n    Args:\n        mountpoint (str): The location of the mount point\n\n    Returns:\n        str: The results of the hdutil detach command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' macpackage.unmount /dev/disk2\n    "
    cmd = f'hdiutil detach "{mountpoint}"'
    return __salt__['cmd.run'](cmd)

def installed_pkgs():
    if False:
        print('Hello World!')
    "\n    Return the list of installed packages on the machine\n\n    Returns:\n        list: List of installed packages\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' macpackage.installed_pkgs\n    "
    cmd = 'pkgutil --pkgs'
    return __salt__['cmd.run'](cmd).split('\n')

def get_pkg_id(pkg):
    if False:
        print('Hello World!')
    "\n    Attempt to get the package ID from a .pkg file\n\n    Args:\n        pkg (str): The location of the pkg file\n\n    Returns:\n        list: List of all of the package IDs\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' macpackage.get_pkg_id /tmp/test.pkg\n    "
    pkg = shlex.quote(pkg)
    package_ids = []
    temp_dir = __salt__['temp.dir'](prefix='pkg-')
    try:
        cmd = f'xar -t -f {pkg} | grep PackageInfo'
        out = __salt__['cmd.run'](cmd, python_shell=True, output_loglevel='quiet')
        files = out.split('\n')
        if 'Error opening' not in out:
            cmd = 'xar -x -f {} {}'.format(pkg, ' '.join(files))
            __salt__['cmd.run'](cmd, cwd=temp_dir, output_loglevel='quiet')
            for f in files:
                i = _get_pkg_id_from_pkginfo(os.path.join(temp_dir, f))
                if i:
                    package_ids.extend(i)
        else:
            package_ids = _get_pkg_id_dir(pkg)
    finally:
        __salt__['file.remove'](temp_dir)
    return package_ids

def get_mpkg_ids(mpkg):
    if False:
        print('Hello World!')
    "\n    Attempt to get the package IDs from a mounted .mpkg file\n\n    Args:\n        mpkg (str): The location of the mounted mpkg file\n\n    Returns:\n        list: List of package IDs\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' macpackage.get_mpkg_ids /dev/disk2\n    "
    mpkg = shlex.quote(mpkg)
    package_infos = []
    base_path = os.path.dirname(mpkg)
    cmd = f'find {base_path} -name *.pkg'
    out = __salt__['cmd.run'](cmd, python_shell=True)
    pkg_files = out.split('\n')
    for p in pkg_files:
        package_infos.extend(get_pkg_id(p))
    return package_infos

def _get_pkg_id_from_pkginfo(pkginfo):
    if False:
        while True:
            i = 10
    pkginfo = shlex.quote(pkginfo)
    cmd = 'cat {} | grep -Eo \'identifier="[a-zA-Z.0-9\\-]*"\' | cut -c 13- | tr -d \'"\''.format(pkginfo)
    out = __salt__['cmd.run'](cmd, python_shell=True)
    if 'No such file' not in out:
        return out.split('\n')
    return []

def _get_pkg_id_dir(path):
    if False:
        while True:
            i = 10
    path = shlex.quote(os.path.join(path, 'Contents/Info.plist'))
    cmd = f'/usr/libexec/PlistBuddy -c "print :CFBundleIdentifier" {path}'
    python_shell = False
    if '*.' in cmd:
        python_shell = True
    out = __salt__['cmd.run'](cmd, python_shell=python_shell)
    if 'Does Not Exist' not in out:
        return [out]
    return []