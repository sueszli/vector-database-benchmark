"""
Install pkg, dmg and .app applications on macOS minions.

"""

import logging
import os
import shlex

import salt.utils.platform

log = logging.getLogger(__name__)

__virtualname__ = "macpackage"


def __virtual__():
    """
    Only work on Mac OS
    """
    if salt.utils.platform.is_darwin():
        return __virtualname__
    return (False, "Only available on Mac OS systems with pipes")


def install(pkg, target="LocalSystem", store=False, allow_untrusted=False):
    """
    Install a pkg file

    Args:
        pkg (str): The package to install
        target (str): The target in which to install the package to
        store (bool): Should the package be installed as if it was from the
                      store?
        allow_untrusted (bool): Allow the installation of untrusted packages?

    Returns:
        dict: A dictionary containing the results of the installation

    CLI Example:

    .. code-block:: bash

        salt '*' macpackage.install test.pkg
    """
    if "*." not in pkg:
        # If we use wildcards, we cannot use quotes
        pkg = shlex.quote(pkg)

    target = shlex.quote(target)

    cmd = f"installer -pkg {pkg} -target {target}"
    if store:
        cmd += " -store"
    if allow_untrusted:
        cmd += " -allowUntrusted"

    # We can only use wildcards in python_shell which is
    # sent by the macpackage state
    python_shell = False
    if "*." in cmd:
        python_shell = True

    return __salt__["cmd.run_all"](cmd, python_shell=python_shell)


def install_app(app, target="/Applications/"):
    """
    Install an app file by moving it into the specified Applications directory

    Args:
        app (str): The location of the .app file
        target (str): The target in which to install the package to
                      Default is ''/Applications/''

    Returns:
        str: The results of the rsync command

    CLI Example:

    .. code-block:: bash

        salt '*' macpackage.install_app /tmp/tmp.app /Applications/
    """

    if target[-4:] != ".app":
        if app[-1:] == "/":
            base_app = os.path.basename(app[:-1])
        else:
            base_app = os.path.basename(app)

        target = os.path.join(target, base_app)

    if not app[-1] == "/":
        app += "/"

    cmd = f'rsync -a --delete "{app}" "{target}"'
    return __salt__["cmd.run"](cmd)


def uninstall_app(app):
    """
    Uninstall an app file by removing it from the Applications directory

    Args:
        app (str): The location of the .app file

    Returns:
        bool: True if successful, otherwise False

    CLI Example:

    .. code-block:: bash

        salt '*' macpackage.uninstall_app /Applications/app.app
    """

    return __salt__["file.remove"](app)


def mount(dmg):
    """
    Attempt to mount a dmg file to a temporary location and return the
    location of the pkg file inside

    Args:
        dmg (str): The location of the dmg file to mount

    Returns:
        tuple: Tuple containing the results of the command along with the mount
               point

    CLI Example:

    .. code-block:: bash

        salt '*' macpackage.mount /tmp/software.dmg
    """

    temp_dir = __salt__["temp.dir"](prefix="dmg-")

    cmd = f'hdiutil attach -readonly -nobrowse -mountpoint {temp_dir} "{dmg}"'

    return __salt__["cmd.run"](cmd), temp_dir


def unmount(mountpoint):
    """
    Attempt to unmount a dmg file from a temporary location

    Args:
        mountpoint (str): The location of the mount point

    Returns:
        str: The results of the hdutil detach command

    CLI Example:

    .. code-block:: bash

        salt '*' macpackage.unmount /dev/disk2
    """

    cmd = f'hdiutil detach "{mountpoint}"'

    return __salt__["cmd.run"](cmd)


def installed_pkgs():
    """
    Return the list of installed packages on the machine

    Returns:
        list: List of installed packages

    CLI Example:

    .. code-block:: bash

        salt '*' macpackage.installed_pkgs
    """

    cmd = "pkgutil --pkgs"

    return __salt__["cmd.run"](cmd).split("\n")


def get_pkg_id(pkg):
    """
    Attempt to get the package ID from a .pkg file

    Args:
        pkg (str): The location of the pkg file

    Returns:
        list: List of all of the package IDs

    CLI Example:

    .. code-block:: bash

        salt '*' macpackage.get_pkg_id /tmp/test.pkg
    """
    pkg = shlex.quote(pkg)
    package_ids = []

    # Create temp directory
    temp_dir = __salt__["temp.dir"](prefix="pkg-")

    try:
        # List all of the PackageInfo files
        cmd = f"xar -t -f {pkg} | grep PackageInfo"
        out = __salt__["cmd.run"](cmd, python_shell=True, output_loglevel="quiet")
        files = out.split("\n")

        if "Error opening" not in out:
            # Extract the PackageInfo files
            cmd = "xar -x -f {} {}".format(pkg, " ".join(files))
            __salt__["cmd.run"](cmd, cwd=temp_dir, output_loglevel="quiet")

            # Find our identifiers
            for f in files:
                i = _get_pkg_id_from_pkginfo(os.path.join(temp_dir, f))
                if i:
                    package_ids.extend(i)
        else:
            package_ids = _get_pkg_id_dir(pkg)

    finally:
        # Clean up
        __salt__["file.remove"](temp_dir)

    return package_ids


def get_mpkg_ids(mpkg):
    """
    Attempt to get the package IDs from a mounted .mpkg file

    Args:
        mpkg (str): The location of the mounted mpkg file

    Returns:
        list: List of package IDs

    CLI Example:

    .. code-block:: bash

        salt '*' macpackage.get_mpkg_ids /dev/disk2
    """
    mpkg = shlex.quote(mpkg)
    package_infos = []
    base_path = os.path.dirname(mpkg)

    # List all of the .pkg files
    cmd = f"find {base_path} -name *.pkg"
    out = __salt__["cmd.run"](cmd, python_shell=True)

    pkg_files = out.split("\n")
    for p in pkg_files:
        package_infos.extend(get_pkg_id(p))

    return package_infos


def _get_pkg_id_from_pkginfo(pkginfo):
    # Find our identifiers
    pkginfo = shlex.quote(pkginfo)
    cmd = "cat {} | grep -Eo 'identifier=\"[a-zA-Z.0-9\\-]*\"' | cut -c 13- | tr -d '\"'".format(
        pkginfo
    )
    out = __salt__["cmd.run"](cmd, python_shell=True)

    if "No such file" not in out:
        return out.split("\n")

    return []


def _get_pkg_id_dir(path):
    path = shlex.quote(os.path.join(path, "Contents/Info.plist"))
    cmd = f'/usr/libexec/PlistBuddy -c "print :CFBundleIdentifier" {path}'

    # We can only use wildcards in python_shell which is
    # sent by the macpackage state
    python_shell = False
    if "*." in cmd:
        python_shell = True

    out = __salt__["cmd.run"](cmd, python_shell=python_shell)

    if "Does Not Exist" not in out:
        return [out]

    return []
