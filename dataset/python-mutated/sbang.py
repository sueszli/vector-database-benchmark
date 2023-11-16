import filecmp
import os
import re
import shutil
import stat
import sys
import tempfile
import llnl.util.filesystem as fs
import llnl.util.tty as tty
import spack.error
import spack.package_prefs
import spack.paths
import spack.spec
import spack.store
if sys.platform == 'darwin':
    system_shebang_limit = 511
else:
    system_shebang_limit = 127
if sys.platform != 'win32':
    import grp
spack_shebang_limit = 4096
interpreter_regex = re.compile(b'#![ \t]*?([^ \t\x00\n]+)')

def sbang_install_path():
    if False:
        for i in range(10):
            print('nop')
    "Location sbang should be installed within Spack's ``install_tree``."
    sbang_root = str(spack.store.STORE.unpadded_root)
    install_path = os.path.join(sbang_root, 'bin', 'sbang')
    path_length = len(install_path)
    if path_length > system_shebang_limit:
        msg = 'Install tree root is too long. Spack cannot patch shebang lines when script path length ({0}) exceeds limit ({1}).\n  {2}'
        msg = msg.format(path_length, system_shebang_limit, install_path)
        raise SbangPathError(msg)
    return install_path

def sbang_shebang_line():
    if False:
        for i in range(10):
            print('nop')
    'Full shebang line that should be prepended to files to use sbang.\n\n    The line returned does not have a final newline (caller should add it\n    if needed).\n\n    This should be the only place in Spack that knows about what\n    interpreter we use for ``sbang``.\n    '
    return '#!/bin/sh %s' % sbang_install_path()

def get_interpreter(binary_string):
    if False:
        print('Hello World!')
    match = interpreter_regex.match(binary_string)
    return None if match is None else match.group(1)

def filter_shebang(path):
    if False:
        i = 10
        return i + 15
    '\n    Adds a second shebang line, using sbang, at the beginning of a file, if necessary.\n    Note: Spack imposes a relaxed shebang line limit, meaning that a newline or end of\n    file must occur before ``spack_shebang_limit`` bytes. If not, the file is not\n    patched.\n    '
    with open(path, 'rb') as original:
        old_shebang_line = original.read(2)
        if old_shebang_line != b'#!':
            return False
        old_shebang_line += original.readline(spack_shebang_limit - 2)
        if len(old_shebang_line) <= system_shebang_limit:
            return False
        if len(old_shebang_line) == spack_shebang_limit and old_shebang_line[-1] != b'\n':
            return False
        new_sbang_line = (sbang_shebang_line() + '\n').encode('utf-8')
        if old_shebang_line == new_sbang_line:
            return
        interpreter = get_interpreter(old_shebang_line)
        if not interpreter:
            return False
        saved_mode = os.stat(path).st_mode
        if not os.access(path, os.W_OK):
            os.chmod(path, saved_mode | stat.S_IWUSR)
        patched = tempfile.NamedTemporaryFile('wb', delete=False)
        patched.write(new_sbang_line)
        if interpreter[-4:] == b'/lua' or interpreter[-7:] == b'/luajit':
            patched.write(b'--!' + old_shebang_line[2:])
        elif interpreter[-5:] == b'/node':
            patched.write(b'//!' + old_shebang_line[2:])
        elif interpreter[-4:] == b'/php':
            patched.write(b'<?php ' + old_shebang_line + b' ?>')
        else:
            patched.write(old_shebang_line)
        shutil.copyfileobj(original, patched)
    patched.close()
    shutil.move(patched.name, path)
    os.chmod(path, saved_mode)
    return True

def filter_shebangs_in_directory(directory, filenames=None):
    if False:
        i = 10
        return i + 15
    if filenames is None:
        filenames = os.listdir(directory)
    is_exe = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    for file in filenames:
        path = os.path.join(directory, file)
        try:
            st = os.lstat(path)
        except (IOError, OSError):
            continue
        if stat.S_ISLNK(st.st_mode) or stat.S_ISDIR(st.st_mode) or (not st.st_mode & is_exe):
            continue
        if filter_shebang(path):
            tty.debug('Patched overlong shebang in %s' % path)

def install_sbang():
    if False:
        return 10
    "Ensure that ``sbang`` is installed in the root of Spack's install_tree.\n\n    This is the shortest known publicly accessible path, and installing\n    ``sbang`` here ensures that users can access the script and that\n    ``sbang`` itself is in a short path.\n    "
    sbang_path = sbang_install_path()
    if os.path.exists(sbang_path) and filecmp.cmp(spack.paths.sbang_script, sbang_path):
        return
    sbang_bin_dir = os.path.dirname(sbang_path)
    fs.mkdirp(sbang_bin_dir)
    group_name = spack.package_prefs.get_package_group(spack.spec.Spec('all'))
    config_mode = spack.package_prefs.get_package_dir_permissions(spack.spec.Spec('all'))
    if group_name:
        os.chmod(sbang_bin_dir, config_mode)
    else:
        fs.set_install_permissions(sbang_bin_dir)
    if group_name and os.stat(sbang_bin_dir).st_gid != grp.getgrnam(group_name).gr_gid:
        os.chown(sbang_bin_dir, os.stat(sbang_bin_dir).st_uid, grp.getgrnam(group_name).gr_gid)
    sbang_tmp_path = os.path.join(os.path.dirname(sbang_path), '.%s.tmp' % os.path.basename(sbang_path))
    shutil.copy(spack.paths.sbang_script, sbang_tmp_path)
    os.chmod(sbang_tmp_path, config_mode)
    if group_name:
        os.chown(sbang_tmp_path, os.stat(sbang_tmp_path).st_uid, grp.getgrnam(group_name).gr_gid)
    os.rename(sbang_tmp_path, sbang_path)

def post_install(spec, explicit=None):
    if False:
        return 10
    'This hook edits scripts so that they call /bin/bash\n    $spack_prefix/bin/sbang instead of something longer than the\n    shebang limit.\n    '
    if spec.external:
        tty.debug('SKIP: shebang filtering [external package]')
        return
    install_sbang()
    for (directory, _, filenames) in os.walk(spec.prefix):
        filter_shebangs_in_directory(directory, filenames)

class SbangPathError(spack.error.SpackError):
    """Raised when the install tree root is too long for sbang to work."""