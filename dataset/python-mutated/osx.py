"""
Utils for Mac OS platform.
"""
import math
import os
import pathlib
import subprocess
import shutil
import tempfile
from macholib.mach_o import LC_BUILD_VERSION, LC_CODE_SIGNATURE, LC_ID_DYLIB, LC_LOAD_DYLIB, LC_LOAD_UPWARD_DYLIB, LC_LOAD_WEAK_DYLIB, LC_PREBOUND_DYLIB, LC_REEXPORT_DYLIB, LC_RPATH, LC_SEGMENT_64, LC_SYMTAB, LC_VERSION_MIN_MACOSX
from macholib.MachO import MachO
import macholib.util
import PyInstaller.log as logging
from PyInstaller import compat
logger = logging.getLogger(__name__)

def is_homebrew_env():
    if False:
        print('Hello World!')
    "\n    Check if Python interpreter was installed via Homebrew command 'brew'.\n\n    :return: True if Homebrew else otherwise.\n    "
    env_prefix = get_homebrew_prefix()
    if env_prefix and compat.base_prefix.startswith(env_prefix):
        return True
    return False

def is_macports_env():
    if False:
        while True:
            i = 10
    "\n    Check if Python interpreter was installed via Macports command 'port'.\n\n    :return: True if Macports else otherwise.\n    "
    env_prefix = get_macports_prefix()
    if env_prefix and compat.base_prefix.startswith(env_prefix):
        return True
    return False

def get_homebrew_prefix():
    if False:
        for i in range(10):
            print('nop')
    '\n    :return: Root path of the Homebrew environment.\n    '
    prefix = shutil.which('brew')
    prefix = os.path.dirname(os.path.dirname(prefix))
    return prefix

def get_macports_prefix():
    if False:
        return 10
    '\n    :return: Root path of the Macports environment.\n    '
    prefix = shutil.which('port')
    prefix = os.path.dirname(os.path.dirname(prefix))
    return prefix

def _find_version_cmd(header):
    if False:
        print('Hello World!')
    '\n    Helper that finds the version command in the given MachO header.\n    '
    version_cmd = [cmd for cmd in header.commands if cmd[0].cmd in {LC_BUILD_VERSION, LC_VERSION_MIN_MACOSX}]
    assert len(version_cmd) == 1, 'Expected exactly one LC_BUILD_VERSION or LC_VERSION_MIN_MACOSX command!'
    return version_cmd[0]

def get_macos_sdk_version(filename):
    if False:
        i = 10
        return i + 15
    '\n    Obtain the version of macOS SDK against which the given binary was built.\n\n    NOTE: currently, version is retrieved only from the first arch slice in the binary.\n\n    :return: (major, minor, revision) tuple\n    '
    binary = MachO(filename)
    header = binary.headers[0]
    version_cmd = _find_version_cmd(header)
    return _hex_triplet(version_cmd[1].sdk)

def _hex_triplet(version):
    if False:
        return 10
    major = (version & 16711680) >> 16
    minor = (version & 65280) >> 8
    revision = version & 255
    return (major, minor, revision)

def macosx_version_min(filename: str) -> tuple:
    if False:
        while True:
            i = 10
    '\n    Get the -macosx-version-min used to compile a macOS binary.\n\n    For fat binaries, the minimum version is selected.\n    '
    versions = []
    for header in MachO(filename).headers:
        cmd = _find_version_cmd(header)
        if cmd[0].cmd == LC_VERSION_MIN_MACOSX:
            versions.append(cmd[1].version)
        else:
            versions.append(cmd[1].minos)
    return min(map(_hex_triplet, versions))

def set_macos_sdk_version(filename, major, minor, revision):
    if False:
        i = 10
        return i + 15
    '\n    Overwrite the macOS SDK version declared in the given binary with the specified version.\n\n    NOTE: currently, only version in the first arch slice is modified.\n    '
    assert 0 <= major <= 255, 'Invalid major version value!'
    assert 0 <= minor <= 255, 'Invalid minor version value!'
    assert 0 <= revision <= 255, 'Invalid revision value!'
    binary = MachO(filename)
    header = binary.headers[0]
    version_cmd = _find_version_cmd(header)
    version_cmd[1].sdk = major << 16 | minor << 8 | revision
    with open(binary.filename, 'rb+') as fp:
        binary.write(fp)

def fix_exe_for_code_signing(filename):
    if False:
        print('Hello World!')
    "\n    Fixes the Mach-O headers to make code signing possible.\n\n    Code signing on Mac OS does not work out of the box with embedding .pkg archive into the executable.\n\n    The fix is done this way:\n    - Make the embedded .pkg archive part of the Mach-O 'String Table'. 'String Table' is at end of the Mac OS exe file,\n      so just change the size of the table to cover the end of the file.\n    - Fix the size of the __LINKEDIT segment.\n\n    Note: the above fix works only if the single-arch thin executable or the last arch slice in a multi-arch fat\n    executable is not signed, because LC_CODE_SIGNATURE comes after LC_SYMTAB, and because modification of headers\n    invalidates the code signature. On modern arm64 macOS, code signature is mandatory, and therefore compilers\n    create a dummy signature when executable is built. In such cases, that signature needs to be removed before this\n    function is called.\n\n    Mach-O format specification: http://developer.apple.com/documentation/Darwin/Reference/ManPages/man5/Mach-O.5.html\n    "
    file_size = os.path.getsize(filename)
    executable = MachO(filename)
    header = executable.headers[-1]
    sign_sec = [cmd for cmd in header.commands if cmd[0].cmd == LC_CODE_SIGNATURE]
    assert len(sign_sec) == 0, 'Executable contains code signature!'
    __LINKEDIT_NAME = b'__LINKEDIT\x00\x00\x00\x00\x00\x00'
    linkedit_seg = [cmd for cmd in header.commands if cmd[0].cmd == LC_SEGMENT_64 and cmd[1].segname == __LINKEDIT_NAME]
    assert len(linkedit_seg) == 1, 'Expected exactly one __LINKEDIT segment!'
    linkedit_seg = linkedit_seg[0][1]
    symtab_sec = [cmd for cmd in header.commands if cmd[0].cmd == LC_SYMTAB]
    assert len(symtab_sec) == 1, 'Expected exactly one SYMTAB section!'
    symtab_sec = symtab_sec[0][1]
    symtab_sec.strsize = file_size - (header.offset + symtab_sec.stroff)
    linkedit_seg.filesize = file_size - (header.offset + linkedit_seg.fileoff)
    page_size = 16384 if _get_arch_string(header.header).startswith('arm64') else 4096
    linkedit_seg.vmsize = math.ceil(linkedit_seg.filesize / page_size) * page_size
    with open(filename, 'rb+') as fp:
        executable.write(fp)
    if executable.fat:
        from macholib.mach_o import FAT_MAGIC, FAT_MAGIC_64, fat_arch, fat_arch64, fat_header
        with open(filename, 'rb+') as fp:
            fat = fat_header.from_fileobj(fp)
            if fat.magic == FAT_MAGIC:
                archs = [fat_arch.from_fileobj(fp) for i in range(fat.nfat_arch)]
            elif fat.magic == FAT_MAGIC_64:
                archs = [fat_arch64.from_fileobj(fp) for i in range(fat.nfat_arch)]
            arch = archs[-1]
            arch.size = file_size - arch.offset
            fp.seek(0)
            fat.to_fileobj(fp)
            for arch in archs:
                arch.to_fileobj(fp)

def _get_arch_string(header):
    if False:
        while True:
            i = 10
    '\n    Converts cputype and cpusubtype from mach_o.mach_header_64 into arch string comparible with lipo/codesign.\n    The list of supported architectures can be found in man(1) arch.\n    '
    cputype = header.cputype
    cpusubtype = header.cpusubtype & 268435455
    if cputype == 16777216 | 7:
        if cpusubtype == 8:
            return 'x86_64h'
        else:
            return 'x86_64'
    elif cputype == 16777216 | 12:
        if cpusubtype == 2:
            return 'arm64e'
        else:
            return 'arm64'
    elif cputype == 7:
        return 'i386'
    assert False, 'Unhandled architecture!'

class InvalidBinaryError(Exception):
    """
    Exception raised by ˙get_binary_architectures˙ when it is passed an invalid binary.
    """
    pass

class IncompatibleBinaryArchError(Exception):
    """
    Exception raised by `binary_to_target_arch` when the passed binary fails the strict architecture check.
    """
    pass

def get_binary_architectures(filename):
    if False:
        for i in range(10):
            print('nop')
    '\n    Inspects the given binary and returns tuple (is_fat, archs), where is_fat is boolean indicating fat/thin binary,\n    and arch is list of architectures with lipo/codesign compatible names.\n    '
    try:
        executable = MachO(filename)
    except ValueError as e:
        raise InvalidBinaryError('Invalid Mach-O binary!') from e
    return (bool(executable.fat), [_get_arch_string(hdr.header) for hdr in executable.headers])

def convert_binary_to_thin_arch(filename, thin_arch, output_filename=None):
    if False:
        i = 10
        return i + 15
    '\n    Convert the given fat binary into thin one with the specified target architecture.\n    '
    output_filename = output_filename or filename
    cmd_args = ['lipo', '-thin', thin_arch, filename, '-output', output_filename]
    p = subprocess.run(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if p.returncode:
        raise SystemError(f'lipo command ({cmd_args}) failed with error code {p.returncode}!\noutput: {p.stdout}')

def merge_into_fat_binary(output_filename, *slice_filenames):
    if False:
        print('Hello World!')
    '\n    Merge the given single-arch thin binary files into a fat binary.\n    '
    cmd_args = ['lipo', '-create', '-output', output_filename, *slice_filenames]
    p = subprocess.run(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if p.returncode:
        raise SystemError(f'lipo command ({cmd_args}) failed with error code {p.returncode}!\noutput: {p.stdout}')

def binary_to_target_arch(filename, target_arch, display_name=None):
    if False:
        print('Hello World!')
    '\n    Check that the given binary contains required architecture slice(s) and convert the fat binary into thin one,\n    if necessary.\n    '
    if not display_name:
        display_name = filename
    (is_fat, archs) = get_binary_architectures(filename)
    if target_arch == 'universal2':
        if not is_fat:
            raise IncompatibleBinaryArchError(f'{display_name} is not a fat binary!')
    elif is_fat:
        if target_arch not in archs:
            raise IncompatibleBinaryArchError(f'{display_name} does not contain slice for {target_arch}!')
        logger.debug('Converting fat binary %s (%s) to thin binary (%s)', filename, display_name, target_arch)
        convert_binary_to_thin_arch(filename, target_arch)
    elif target_arch not in archs:
        raise IncompatibleBinaryArchError(f'{display_name} is incompatible with target arch {target_arch} (has arch: {archs[0]})!')

def remove_signature_from_binary(filename):
    if False:
        print('Hello World!')
    '\n    Remove the signature from all architecture slices of the given binary file using the codesign utility.\n    '
    logger.debug('Removing signature from file %r', filename)
    cmd_args = ['codesign', '--remove', '--all-architectures', filename]
    p = subprocess.run(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if p.returncode:
        raise SystemError(f'codesign command ({cmd_args}) failed with error code {p.returncode}!\noutput: {p.stdout}')

def sign_binary(filename, identity=None, entitlements_file=None, deep=False):
    if False:
        return 10
    '\n    Sign the binary using codesign utility. If no identity is provided, ad-hoc signing is performed.\n    '
    extra_args = []
    if not identity:
        identity = '-'
    else:
        extra_args.append('--options=runtime')
    if entitlements_file:
        extra_args.append('--entitlements')
        extra_args.append(entitlements_file)
    if deep:
        extra_args.append('--deep')
    logger.debug('Signing file %r', filename)
    cmd_args = ['codesign', '-s', identity, '--force', '--all-architectures', '--timestamp', *extra_args, filename]
    p = subprocess.run(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if p.returncode:
        raise SystemError(f'codesign command ({cmd_args}) failed with error code {p.returncode}!\noutput: {p.stdout}')

def set_dylib_dependency_paths(filename, target_rpath):
    if False:
        while True:
            i = 10
    "\n    Modify the given dylib's identity (in LC_ID_DYLIB command) and the paths to dependent dylibs (in LC_LOAD_DYLIB)\n    commands into `@rpath/<basename>` format, remove any existing rpaths (LC_RPATH commands), and add a new rpath\n    (LC_RPATH command) with the specified path.\n\n    Uses `install-tool-name` utility to make the changes.\n\n    The system libraries (e.g., the ones found in /usr/lib) are exempted from path rewrite.\n\n    For multi-arch fat binaries, this function extracts each slice into temporary file, processes it separately,\n    and then merges all processed slices back into fat binary. This is necessary because `install-tool-name` cannot\n    modify rpaths in cases when an existing rpath is present only in one slice.\n    "
    (is_fat, archs) = get_binary_architectures(filename)
    if is_fat:
        with tempfile.TemporaryDirectory() as tmpdir:
            slice_filenames = []
            for arch in archs:
                slice_filename = os.path.join(tmpdir, arch)
                convert_binary_to_thin_arch(filename, arch, output_filename=slice_filename)
                _set_dylib_dependency_paths(slice_filename, target_rpath)
                slice_filenames.append(slice_filename)
            merge_into_fat_binary(filename, *slice_filenames)
    else:
        _set_dylib_dependency_paths(filename, target_rpath)

def _set_dylib_dependency_paths(filename, target_rpath):
    if False:
        for i in range(10):
            print('nop')
    '\n    The actual implementation of set_dylib_dependency_paths functionality.\n\n    Implicitly assumes that a single-arch thin binary is given.\n    '
    _RELOCATABLE = {LC_LOAD_DYLIB, LC_LOAD_UPWARD_DYLIB, LC_LOAD_WEAK_DYLIB, LC_PREBOUND_DYLIB, LC_REEXPORT_DYLIB}
    binary = MachO(filename)
    dylib_id = None
    rpaths = set()
    linked_libs = set()
    for header in binary.headers:
        for cmd in header.commands:
            lc_type = cmd[0].cmd
            if lc_type not in _RELOCATABLE and lc_type not in {LC_RPATH, LC_ID_DYLIB}:
                continue
            path = cmd[2].decode('utf-8').rstrip('\x00')
            if lc_type in _RELOCATABLE:
                linked_libs.add(path)
            elif lc_type == LC_RPATH:
                rpaths.add(path)
            elif lc_type == LC_ID_DYLIB:
                dylib_id = path
    del binary
    normalized_dylib_id = None
    if dylib_id:
        normalized_dylib_id = str(pathlib.PurePath('@rpath') / pathlib.PurePath(dylib_id).name)
    changed_lib_paths = []
    rpath_required = False
    for linked_lib in linked_libs:
        if macholib.util.in_system_path(linked_lib):
            continue
        _exemptions = ['/Library/Frameworks/Tcl.framework/', '/Library/Frameworks/Tk.framework/']
        if any([x in linked_lib for x in _exemptions]):
            continue
        rpath_required = True
        new_path = str(pathlib.PurePath('@rpath') / pathlib.PurePath(linked_lib).name)
        if linked_lib == new_path:
            continue
        changed_lib_paths.append((linked_lib, new_path))
    install_name_tool_args = []
    if normalized_dylib_id and normalized_dylib_id != dylib_id:
        install_name_tool_args += ['-id', normalized_dylib_id]
    for (original_path, new_path) in changed_lib_paths:
        install_name_tool_args += ['-change', original_path, new_path]
    for rpath in rpaths:
        if rpath == target_rpath:
            continue
        install_name_tool_args += ['-delete_rpath', rpath]
    if rpath_required and target_rpath not in rpaths:
        install_name_tool_args += ['-add_rpath', target_rpath]
    if not install_name_tool_args:
        return
    cmd_args = ['install_name_tool', *install_name_tool_args, filename]
    p = subprocess.run(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if p.returncode:
        raise SystemError(f'install_name_tool command ({cmd_args}) failed with error code {p.returncode}!\noutput: {p.stdout}')

def is_framework_bundle_lib(lib_path):
    if False:
        i = 10
        return i + 15
    '\n    Check if the given shared library is part of a .framework bundle.\n    '
    lib_path = pathlib.PurePath(lib_path)
    if lib_path.parent.parent.name != 'Versions':
        return False
    if lib_path.parent.parent.parent.name != lib_path.name + '.framework':
        return False
    return True

def collect_files_from_framework_bundles(collected_files):
    if False:
        print('Hello World!')
    "\n    Scan the given TOC list of collected files for shared libraries that are collected from macOS .framework bundles,\n    and collect the bundles' Info.plist files. Additionally, the following symbolic links:\n      - `Versions/Current` pointing to the `Versions/<version>` directory containing the binary\n      - `<name>` in the top-level .framework directory, pointing to `Versions/Current/<name>`\n      - `Resources` in the top-level .framework directory, pointing to `Versions/Current/Resources`\n      - additional directories in top-level .framework directory, pointing to their counterparts in `Versions/Current`\n        directory.\n\n    Returns TOC list for the discovered Info.plist files and generated symbolic links. The list does not contain\n    duplicated entries.\n    "
    invalid_framework_found = False
    framework_files = set()
    framework_paths = set()
    for (dest_name, src_name, typecode) in collected_files:
        if typecode != 'BINARY':
            continue
        src_path = pathlib.Path(src_name)
        dest_path = pathlib.PurePath(dest_name)
        if not is_framework_bundle_lib(src_path):
            continue
        if not is_framework_bundle_lib(dest_path):
            continue
        info_plist_src = src_path.parent / 'Resources' / 'Info.plist'
        if not info_plist_src.is_file():
            info_plist_src_top = src_path.parent.parent.parent / 'Resources' / 'Info.plist'
            if not info_plist_src_top.is_file():
                invalid_framework_found = True
                framework_dir = src_path.parent.parent.parent
                if compat.strict_collect_mode:
                    raise SystemError(f'Could not find Info.plist in {framework_dir}!')
                else:
                    logger.warning('Could not find Info.plist in %s!', framework_dir)
                    continue
            info_plist_src = info_plist_src_top
        info_plist_dest = dest_path.parent / 'Resources' / 'Info.plist'
        framework_files.add((str(info_plist_dest), str(info_plist_src), 'DATA'))
        framework_files.add((str(dest_path.parent.parent / 'Current'), str(dest_path.parent.name), 'SYMLINK'))
        dest_framework_path = dest_path.parent.parent.parent
        framework_files.add((str(dest_framework_path / dest_path.name), str(pathlib.PurePath('Versions/Current') / dest_path.name), 'SYMLINK'))
        framework_files.add((str(dest_framework_path / 'Resources'), 'Versions/Current/Resources', 'SYMLINK'))
        framework_paths.add(dest_framework_path)
    VALID_SUBDIRS = {'Helpers', 'Resources'}
    for dest_framework_path in framework_paths:
        for (dest_name, src_name, typecode) in collected_files:
            dest_path = pathlib.PurePath(dest_name)
            try:
                remaining_path = dest_path.relative_to(dest_framework_path)
            except ValueError:
                continue
            remaining_path_parts = remaining_path.parts
            if remaining_path_parts[0] != 'Versions':
                continue
            dir_name = remaining_path_parts[2]
            if dir_name not in VALID_SUBDIRS:
                continue
            framework_files.add((str(dest_framework_path / dir_name), str(pathlib.PurePath('Versions/Current') / dir_name), 'SYMLINK'))
    if invalid_framework_found:
        logger.warning('One or more collected .framework bundles have missing Info.plist file. If you are building an .app bundle, you will most likely not be able to code-sign it.')
    return sorted(framework_files)