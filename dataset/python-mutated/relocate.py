import collections
import itertools
import os
import re
from collections import OrderedDict
import macholib.mach_o
import macholib.MachO
import llnl.util.filesystem as fs
import llnl.util.lang
import llnl.util.tty as tty
from llnl.util.lang import memoized
from llnl.util.symlink import symlink
import spack.paths
import spack.platforms
import spack.repo
import spack.spec
import spack.store
import spack.util.elf as elf
import spack.util.executable as executable
from .relocate_text import BinaryFilePrefixReplacer, TextFilePrefixReplacer
is_macos = str(spack.platforms.real_host()) == 'darwin'

class InstallRootStringError(spack.error.SpackError):

    def __init__(self, file_path, root_path):
        if False:
            print('Hello World!')
        "Signal that the relocated binary still has the original\n        Spack's store root string\n\n        Args:\n            file_path (str): path of the binary\n            root_path (str): original Spack's store root string\n        "
        super().__init__('\n %s \ncontains string\n %s \nafter replacing it in rpaths.\nPackage should not be relocated.\n Use -a to override.' % (file_path, root_path))

@memoized
def _patchelf():
    if False:
        i = 10
        return i + 15
    'Return the full path to the patchelf binary, if available, else None.'
    import spack.bootstrap
    if is_macos:
        return None
    with spack.bootstrap.ensure_bootstrap_configuration():
        patchelf = spack.bootstrap.ensure_patchelf_in_path_or_raise()
    return patchelf.path

def _elf_rpaths_for(path):
    if False:
        for i in range(10):
            print('nop')
    'Return the RPATHs for an executable or a library.\n\n    Args:\n        path (str): full path to the executable or library\n\n    Return:\n        RPATHs as a list of strings. Returns an empty array\n        on ELF parsing errors, or when the ELF file simply\n        has no rpaths.\n    '
    return elf.get_rpaths(path) or []

def _make_relative(reference_file, path_root, paths):
    if False:
        for i in range(10):
            print('nop')
    'Return a list where any path in ``paths`` that starts with\n    ``path_root`` is made relative to the directory in which the\n    reference file is stored.\n\n    After a path is made relative it is prefixed with the ``$ORIGIN``\n    string.\n\n    Args:\n        reference_file (str): file from which the reference directory\n            is computed\n        path_root (str): root of the relative paths\n        paths: (list) paths to be examined\n\n    Returns:\n        List of relative paths\n    '
    start_directory = os.path.dirname(reference_file)
    pattern = re.compile(path_root)
    relative_paths = []
    for path in paths:
        if pattern.match(path):
            rel = os.path.relpath(path, start=start_directory)
            path = os.path.join('$ORIGIN', rel)
        relative_paths.append(path)
    return relative_paths

def _normalize_relative_paths(start_path, relative_paths):
    if False:
        for i in range(10):
            print('nop')
    'Normalize the relative paths with respect to the original path name\n    of the file (``start_path``).\n\n    The paths that are passed to this function existed or were relevant\n    on another filesystem, so os.path.abspath cannot be used.\n\n    A relative path may contain the signifier $ORIGIN. Assuming that\n    ``start_path`` is absolute, this implies that the relative path\n    (relative to start_path) should be replaced with an absolute path.\n\n    Args:\n        start_path (str): path from which the starting directory\n            is extracted\n        relative_paths (str): list of relative paths as obtained by a\n            call to :ref:`_make_relative`\n\n    Returns:\n        List of normalized paths\n    '
    normalized_paths = []
    pattern = re.compile(re.escape('$ORIGIN'))
    start_directory = os.path.dirname(start_path)
    for path in relative_paths:
        if path.startswith('$ORIGIN'):
            sub = pattern.sub(start_directory, path)
            path = os.path.normpath(sub)
        normalized_paths.append(path)
    return normalized_paths

def _decode_macho_data(bytestring):
    if False:
        for i in range(10):
            print('nop')
    return bytestring.rstrip(b'\x00').decode('ascii')

def macho_make_paths_relative(path_name, old_layout_root, rpaths, deps, idpath):
    if False:
        print('Hello World!')
    '\n    Return a dictionary mapping the original rpaths to the relativized rpaths.\n    This dictionary is used to replace paths in mach-o binaries.\n    Replace old_dir with relative path from dirname of path name\n    in rpaths and deps; idpath is replaced with @rpath/libname.\n    '
    paths_to_paths = dict()
    if idpath:
        paths_to_paths[idpath] = os.path.join('@rpath', '%s' % os.path.basename(idpath))
    for rpath in rpaths:
        if re.match(old_layout_root, rpath):
            rel = os.path.relpath(rpath, start=os.path.dirname(path_name))
            paths_to_paths[rpath] = os.path.join('@loader_path', '%s' % rel)
        else:
            paths_to_paths[rpath] = rpath
    for dep in deps:
        if re.match(old_layout_root, dep):
            rel = os.path.relpath(dep, start=os.path.dirname(path_name))
            paths_to_paths[dep] = os.path.join('@loader_path', '%s' % rel)
        else:
            paths_to_paths[dep] = dep
    return paths_to_paths

def macho_make_paths_normal(orig_path_name, rpaths, deps, idpath):
    if False:
        return 10
    "\n    Return a dictionary mapping the relativized rpaths to the original rpaths.\n    This dictionary is used to replace paths in mach-o binaries.\n    Replace '@loader_path' with the dirname of the origname path name\n    in rpaths and deps; idpath is replaced with the original path name\n    "
    rel_to_orig = dict()
    if idpath:
        rel_to_orig[idpath] = orig_path_name
    for rpath in rpaths:
        if re.match('@loader_path', rpath):
            norm = os.path.normpath(re.sub(re.escape('@loader_path'), os.path.dirname(orig_path_name), rpath))
            rel_to_orig[rpath] = norm
        else:
            rel_to_orig[rpath] = rpath
    for dep in deps:
        if re.match('@loader_path', dep):
            norm = os.path.normpath(re.sub(re.escape('@loader_path'), os.path.dirname(orig_path_name), dep))
            rel_to_orig[dep] = norm
        else:
            rel_to_orig[dep] = dep
    return rel_to_orig

def macho_find_paths(orig_rpaths, deps, idpath, old_layout_root, prefix_to_prefix):
    if False:
        print('Hello World!')
    '\n    Inputs\n    original rpaths from mach-o binaries\n    dependency libraries for mach-o binaries\n    id path of mach-o libraries\n    old install directory layout root\n    prefix_to_prefix dictionary which maps prefixes in the old directory layout\n    to directories in the new directory layout\n    Output\n    paths_to_paths dictionary which maps all of the old paths to new paths\n    '
    paths_to_paths = dict()
    for orig_rpath in orig_rpaths:
        if orig_rpath.startswith(old_layout_root):
            for (old_prefix, new_prefix) in prefix_to_prefix.items():
                if orig_rpath.startswith(old_prefix):
                    new_rpath = re.sub(re.escape(old_prefix), new_prefix, orig_rpath)
                    paths_to_paths[orig_rpath] = new_rpath
        else:
            paths_to_paths[orig_rpath] = orig_rpath
    if idpath:
        for (old_prefix, new_prefix) in prefix_to_prefix.items():
            if idpath.startswith(old_prefix):
                paths_to_paths[idpath] = re.sub(re.escape(old_prefix), new_prefix, idpath)
    for dep in deps:
        for (old_prefix, new_prefix) in prefix_to_prefix.items():
            if dep.startswith(old_prefix):
                paths_to_paths[dep] = re.sub(re.escape(old_prefix), new_prefix, dep)
        if dep.startswith('@'):
            paths_to_paths[dep] = dep
    return paths_to_paths

def modify_macho_object(cur_path, rpaths, deps, idpath, paths_to_paths):
    if False:
        print('Hello World!')
    '\n    This function is used to make machO buildcaches on macOS by\n    replacing old paths with new paths using install_name_tool\n    Inputs:\n    mach-o binary to be modified\n    original rpaths\n    original dependency paths\n    original id path if a mach-o library\n    dictionary mapping paths in old install layout to new install layout\n    '
    if 'libgcc_' in cur_path:
        return
    args = []
    if idpath:
        new_idpath = paths_to_paths.get(idpath, None)
        if new_idpath and (not idpath == new_idpath):
            args += [('-id', new_idpath)]
    for dep in deps:
        new_dep = paths_to_paths.get(dep)
        if new_dep and dep != new_dep:
            args += [('-change', dep, new_dep)]
    new_rpaths = []
    for orig_rpath in rpaths:
        new_rpath = paths_to_paths.get(orig_rpath)
        if new_rpath and (not orig_rpath == new_rpath):
            args_to_add = ('-rpath', orig_rpath, new_rpath)
            if args_to_add not in args and new_rpath not in new_rpaths:
                args += [args_to_add]
                new_rpaths.append(new_rpath)
    args = list(itertools.chain.from_iterable(llnl.util.lang.dedupe(args)))
    if args:
        args.append(str(cur_path))
        install_name_tool = executable.Executable('install_name_tool')
        install_name_tool(*args)
    return

def modify_object_macholib(cur_path, paths_to_paths):
    if False:
        print('Hello World!')
    '\n    This function is used when install machO buildcaches on linux by\n    rewriting mach-o loader commands for dependency library paths of\n    mach-o binaries and the id path for mach-o libraries.\n    Rewritting of rpaths is handled by replace_prefix_bin.\n    Inputs\n    mach-o binary to be modified\n    dictionary mapping paths in old install layout to new install layout\n    '
    dll = macholib.MachO.MachO(cur_path)
    dll.rewriteLoadCommands(paths_to_paths.get)
    try:
        f = open(dll.filename, 'rb+')
        for header in dll.headers:
            f.seek(0)
            dll.write(f)
        f.seek(0, 2)
        f.flush()
        f.close()
    except Exception:
        pass
    return

def macholib_get_paths(cur_path):
    if False:
        print('Hello World!')
    'Get rpaths, dependent libraries, and library id of mach-o objects.'
    headers = macholib.MachO.MachO(cur_path).headers
    if not headers:
        tty.warn('Failed to read Mach-O headers: {0}'.format(cur_path))
        commands = []
    else:
        if len(headers) > 1:
            tty.warn('Encountered fat binary: {0}'.format(cur_path))
        if headers[-1].filetype == 'dylib_stub':
            tty.warn('File is a stub, not a full library: {0}'.format(cur_path))
        commands = headers[-1].commands
    LC_ID_DYLIB = macholib.mach_o.LC_ID_DYLIB
    LC_LOAD_DYLIB = macholib.mach_o.LC_LOAD_DYLIB
    LC_RPATH = macholib.mach_o.LC_RPATH
    ident = None
    rpaths = []
    deps = []
    for (load_command, dylib_command, data) in commands:
        cmd = load_command.cmd
        if cmd == LC_RPATH:
            rpaths.append(_decode_macho_data(data))
        elif cmd == LC_LOAD_DYLIB:
            deps.append(_decode_macho_data(data))
        elif cmd == LC_ID_DYLIB:
            ident = _decode_macho_data(data)
    return (rpaths, deps, ident)

def _set_elf_rpaths(target, rpaths):
    if False:
        print('Hello World!')
    'Replace the original RPATH of the target with the paths passed\n    as arguments.\n\n    Args:\n        target: target executable. Must be an ELF object.\n        rpaths: paths to be set in the RPATH\n\n    Returns:\n        A string concatenating the stdout and stderr of the call\n        to ``patchelf`` if it was invoked\n    '
    rpaths_str = ':'.join(rpaths)
    (patchelf, output) = (executable.Executable(_patchelf()), None)
    try:
        patchelf_args = ['--force-rpath', '--set-rpath', rpaths_str, target]
        output = patchelf(*patchelf_args, output=str, error=str)
    except executable.ProcessError as e:
        msg = 'patchelf --force-rpath --set-rpath {0} failed with error {1}'
        tty.warn(msg.format(target, e))
    return output

def needs_binary_relocation(m_type, m_subtype):
    if False:
        print('Hello World!')
    'Returns True if the file with MIME type/subtype passed as arguments\n    needs binary relocation, False otherwise.\n\n    Args:\n        m_type (str): MIME type of the file\n        m_subtype (str): MIME subtype of the file\n    '
    subtypes = ('x-executable', 'x-sharedlib', 'x-mach-binary', 'x-pie-executable')
    if m_type == 'application':
        if m_subtype in subtypes:
            return True
    return False

def needs_text_relocation(m_type, m_subtype):
    if False:
        print('Hello World!')
    'Returns True if the file with MIME type/subtype passed as arguments\n    needs text relocation, False otherwise.\n\n    Args:\n        m_type (str): MIME type of the file\n        m_subtype (str): MIME subtype of the file\n    '
    return m_type == 'text'

def relocate_macho_binaries(path_names, old_layout_root, new_layout_root, prefix_to_prefix, rel, old_prefix, new_prefix):
    if False:
        while True:
            i = 10
    '\n    Use macholib python package to get the rpaths, depedent libraries\n    and library identity for libraries from the MachO object. Modify them\n    with the replacement paths queried from the dictionary mapping old layout\n    prefixes to hashes and the dictionary mapping hashes to the new layout\n    prefixes.\n    '
    for path_name in path_names:
        if path_name.endswith('.o'):
            continue
        if rel:
            (rpaths, deps, idpath) = macholib_get_paths(path_name)
            orig_path_name = re.sub(re.escape(new_prefix), old_prefix, path_name)
            rel_to_orig = macho_make_paths_normal(orig_path_name, rpaths, deps, idpath)
            if is_macos:
                modify_macho_object(path_name, rpaths, deps, idpath, rel_to_orig)
            else:
                modify_object_macholib(path_name, rel_to_orig)
            (rpaths, deps, idpath) = macholib_get_paths(path_name)
            paths_to_paths = macho_find_paths(rpaths, deps, idpath, old_layout_root, prefix_to_prefix)
            if is_macos:
                modify_macho_object(path_name, rpaths, deps, idpath, paths_to_paths)
            else:
                modify_object_macholib(path_name, paths_to_paths)
            (rpaths, deps, idpath) = macholib_get_paths(path_name)
            paths_to_paths = macho_make_paths_relative(path_name, new_layout_root, rpaths, deps, idpath)
            if is_macos:
                modify_macho_object(path_name, rpaths, deps, idpath, paths_to_paths)
            else:
                modify_object_macholib(path_name, paths_to_paths)
        else:
            (rpaths, deps, idpath) = macholib_get_paths(path_name)
            paths_to_paths = macho_find_paths(rpaths, deps, idpath, old_layout_root, prefix_to_prefix)
            if is_macos:
                modify_macho_object(path_name, rpaths, deps, idpath, paths_to_paths)
            else:
                modify_object_macholib(path_name, paths_to_paths)

def _transform_rpaths(orig_rpaths, orig_root, new_prefixes):
    if False:
        print('Hello World!')
    'Return an updated list of RPATHs where each entry in the original list\n    starting with the old root is relocated to another place according to the\n    mapping passed as argument.\n\n    Args:\n        orig_rpaths (list): list of the original RPATHs\n        orig_root (str): original root to be substituted\n        new_prefixes (dict): dictionary that maps the original prefixes to\n            where they should be relocated\n\n    Returns:\n        List of paths\n    '
    new_rpaths = []
    for orig_rpath in orig_rpaths:
        if not orig_rpath.startswith(orig_root):
            new_rpaths.append(orig_rpath)
            continue
        for (old_prefix, new_prefix) in new_prefixes.items():
            if orig_rpath.startswith(old_prefix):
                new_rpath = re.sub(re.escape(old_prefix), new_prefix, orig_rpath)
                if new_rpath not in new_rpaths:
                    new_rpaths.append(new_rpath)
    return new_rpaths

def new_relocate_elf_binaries(binaries, prefix_to_prefix):
    if False:
        while True:
            i = 10
    'Take a list of binaries, and an ordered dictionary of\n    prefix to prefix mapping, and update the rpaths accordingly.'
    prefix_to_prefix = OrderedDict(((k.encode('utf-8'), v.encode('utf-8')) for (k, v) in prefix_to_prefix.items()))
    for path in binaries:
        try:
            elf.replace_rpath_in_place_or_raise(path, prefix_to_prefix)
        except elf.ElfDynamicSectionUpdateFailed as e:
            _set_elf_rpaths(path, e.new.decode('utf-8').split(':'))

def relocate_elf_binaries(binaries, orig_root, new_root, new_prefixes, rel, orig_prefix, new_prefix):
    if False:
        i = 10
        return i + 15
    'Relocate the binaries passed as arguments by changing their RPATHs.\n\n    Use patchelf to get the original RPATHs and then replace them with\n    rpaths in the new directory layout.\n\n    New RPATHs are determined from a dictionary mapping the prefixes in the\n    old directory layout to the prefixes in the new directory layout if the\n    rpath was in the old layout root, i.e. system paths are not replaced.\n\n    Args:\n        binaries (list): list of binaries that might need relocation, located\n            in the new prefix\n        orig_root (str): original root to be substituted\n        new_root (str): new root to be used, only relevant for relative RPATHs\n        new_prefixes (dict): dictionary that maps the original prefixes to\n            where they should be relocated\n        rel (bool): True if the RPATHs are relative, False if they are absolute\n        orig_prefix (str): prefix where the executable was originally located\n        new_prefix (str): prefix where we want to relocate the executable\n    '
    for new_binary in binaries:
        orig_rpaths = _elf_rpaths_for(new_binary)
        if rel:
            orig_binary = re.sub(re.escape(new_prefix), orig_prefix, new_binary)
            orig_norm_rpaths = _normalize_relative_paths(orig_binary, orig_rpaths)
            new_norm_rpaths = _transform_rpaths(orig_norm_rpaths, orig_root, new_prefixes)
            new_rpaths = _make_relative(new_binary, new_root, new_norm_rpaths)
            if sorted(new_rpaths) != sorted(orig_rpaths):
                _set_elf_rpaths(new_binary, new_rpaths)
        else:
            new_rpaths = _transform_rpaths(orig_rpaths, orig_root, new_prefixes)
            _set_elf_rpaths(new_binary, new_rpaths)

def make_link_relative(new_links, orig_links):
    if False:
        for i in range(10):
            print('nop')
    'Compute the relative target from the original link and\n    make the new link relative.\n\n    Args:\n        new_links (list): new links to be made relative\n        orig_links (list): original links\n    '
    for (new_link, orig_link) in zip(new_links, orig_links):
        target = os.readlink(orig_link)
        relative_target = os.path.relpath(target, os.path.dirname(orig_link))
        os.unlink(new_link)
        symlink(relative_target, new_link)

def make_macho_binaries_relative(cur_path_names, orig_path_names, old_layout_root):
    if False:
        while True:
            i = 10
    '\n    Replace old RPATHs with paths relative to old_dir in binary files\n    '
    if not is_macos:
        return
    for (cur_path, orig_path) in zip(cur_path_names, orig_path_names):
        (rpaths, deps, idpath) = macholib_get_paths(cur_path)
        paths_to_paths = macho_make_paths_relative(orig_path, old_layout_root, rpaths, deps, idpath)
        modify_macho_object(cur_path, rpaths, deps, idpath, paths_to_paths)

def make_elf_binaries_relative(new_binaries, orig_binaries, orig_layout_root):
    if False:
        return 10
    'Replace the original RPATHs in the new binaries making them\n    relative to the original layout root.\n\n    Args:\n        new_binaries (list): new binaries whose RPATHs is to be made relative\n        orig_binaries (list): original binaries\n        orig_layout_root (str): path to be used as a base for making\n            RPATHs relative\n    '
    for (new_binary, orig_binary) in zip(new_binaries, orig_binaries):
        orig_rpaths = _elf_rpaths_for(new_binary)
        if orig_rpaths:
            new_rpaths = _make_relative(orig_binary, orig_layout_root, orig_rpaths)
            _set_elf_rpaths(new_binary, new_rpaths)

def warn_if_link_cant_be_relocated(link, target):
    if False:
        return 10
    if not os.path.isabs(target):
        return
    tty.warn('Symbolic link at "{}" to "{}" cannot be relocated'.format(link, target))

def relocate_links(links, prefix_to_prefix):
    if False:
        return 10
    'Relocate links to a new install prefix.'
    regex = re.compile('|'.join((re.escape(p) for p in prefix_to_prefix.keys())))
    for link in links:
        old_target = os.readlink(link)
        match = regex.match(old_target)
        if match is None:
            warn_if_link_cant_be_relocated(link, old_target)
            continue
        new_target = prefix_to_prefix[match.group()] + old_target[match.end():]
        os.unlink(link)
        symlink(new_target, link)

def relocate_text(files, prefixes):
    if False:
        for i in range(10):
            print('nop')
    "Relocate text file from the original installation prefix to the\n    new prefix.\n\n    Relocation also affects the the path in Spack's sbang script.\n\n    Args:\n        files (list): Text files to be relocated\n        prefixes (OrderedDict): String prefixes which need to be changed\n    "
    TextFilePrefixReplacer.from_strings_or_bytes(prefixes).apply(files)

def relocate_text_bin(binaries, prefixes):
    if False:
        return 10
    'Replace null terminated path strings hard-coded into binaries.\n\n    The new install prefix must be shorter than the original one.\n\n    Args:\n        binaries (list): binaries to be relocated\n        prefixes (OrderedDict): String prefixes which need to be changed.\n\n    Raises:\n      spack.relocate_text.BinaryTextReplaceError: when the new path is longer than the old path\n    '
    return BinaryFilePrefixReplacer.from_strings_or_bytes(prefixes).apply(binaries)

def is_binary(filename):
    if False:
        for i in range(10):
            print('nop')
    'Returns true if a file is binary, False otherwise\n\n    Args:\n        filename: file to be tested\n\n    Returns:\n        True or False\n    '
    (m_type, _) = fs.mime_type(filename)
    msg = '[{0}] -> '.format(filename)
    if m_type == 'application':
        tty.debug(msg + 'BINARY FILE')
        return True
    tty.debug(msg + 'TEXT FILE')
    return False

@llnl.util.lang.memoized
def _exists_dir(dirname):
    if False:
        for i in range(10):
            print('nop')
    return os.path.isdir(dirname)

def fixup_macos_rpath(root, filename):
    if False:
        print('Hello World!')
    'Apply rpath fixups to the given file.\n\n    Args:\n        root: absolute path to the parent directory\n        filename: relative path to the library or binary\n\n    Returns:\n        True if fixups were applied, else False\n    '
    abspath = os.path.join(root, filename)
    if fs.mime_type(abspath) != ('application', 'x-mach-binary'):
        return False
    (rpath_list, deps, id_dylib) = macholib_get_paths(abspath)
    add_rpaths = set()
    del_rpaths = set()
    rpaths = collections.defaultdict(int)
    for rpath in rpath_list:
        rpaths[rpath] += 1
    args = []
    spack_root = spack.store.STORE.layout.root
    for name in deps:
        if name.startswith(spack_root):
            tty.debug('Spack-installed dependency for {0}: {1}'.format(abspath, name))
            (dirname, basename) = os.path.split(name)
            if dirname != root or dirname in rpaths:
                args += ['-change', name, '@rpath/' + basename]
                add_rpaths.add(dirname.rstrip('/'))
    for (rpath, count) in rpaths.items():
        if rpath.startswith('@loader_path') or rpath.startswith('@executable_path'):
            pass
        elif not _exists_dir(rpath):
            tty.debug('Nonexistent rpath in {0}: {1}'.format(abspath, rpath))
            del_rpaths.add(rpath)
        elif count > 1:
            tty_debug = tty.debug if count == 2 else tty.warn
            tty_debug('Rpath appears {0} times in {1}: {2}'.format(count, abspath, rpath))
            del_rpaths.add(rpath)
    for rpath in del_rpaths:
        args += ['-delete_rpath', rpath]
    for rpath in add_rpaths - del_rpaths - set(rpaths):
        args += ['-add_rpath', rpath]
    if not args:
        return False
    args.append(abspath)
    executable.Executable('install_name_tool')(*args)
    return True

def fixup_macos_rpaths(spec):
    if False:
        return 10
    'Remove duplicate and nonexistent rpaths.\n\n    Some autotools packages write their own ``-rpath`` entries in addition to\n    those implicitly added by the Spack compiler wrappers. On Linux these\n    duplicate rpaths are eliminated, but on macOS they result in multiple\n    entries which makes it harder to adjust with ``install_name_tool\n    -delete_rpath``.\n    '
    if spec.external or spec.virtual:
        tty.warn('external or virtual package cannot be fixed up: {0!s}'.format(spec))
        return False
    if 'platform=darwin' not in spec:
        raise NotImplementedError('fixup_macos_rpaths requires macOS')
    applied = 0
    libs = frozenset(['lib', 'lib64', 'libexec', 'plugins', 'Library', 'Frameworks'])
    prefix = spec.prefix
    if not os.path.exists(prefix):
        raise RuntimeError('Could not fix up install prefix spec {0} because it does not exist: {1!s}'.format(prefix, spec.name))
    for (root, dirs, files) in os.walk(prefix, topdown=True):
        dirs[:] = set(dirs) & libs
        for name in files:
            try:
                needed_fix = fixup_macos_rpath(root, name)
            except Exception as e:
                tty.warn('Failed to apply library fixups to: {0}/{1}: {2!s}'.format(root, name, e))
                needed_fix = False
            if needed_fix:
                applied += 1
    specname = spec.format('{name}{/hash:7}')
    if applied:
        tty.info('Fixed rpaths for {0:d} {1} installed to {2}'.format(applied, 'binary' if applied == 1 else 'binaries', specname))
    else:
        tty.debug('No rpath fixup needed for ' + specname)