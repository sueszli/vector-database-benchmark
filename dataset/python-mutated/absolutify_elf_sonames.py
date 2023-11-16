import os
import llnl.util.tty as tty
from llnl.util.filesystem import BaseDirectoryVisitor, visit_directory_tree
from llnl.util.lang import elide_list
import spack.bootstrap
import spack.config
import spack.relocate
from spack.util.elf import ElfParsingError, parse_elf
from spack.util.executable import Executable

def is_shared_library_elf(filepath):
    if False:
        print('Hello World!')
    'Return true if filepath is most a shared library.\n    Our definition of a shared library for ELF requires:\n    1. a dynamic section,\n    2. a soname OR lack of interpreter.\n    The problem is that PIE objects (default on Ubuntu) are\n    ET_DYN too, and not all shared libraries have a soname...\n    no interpreter is typically the best indicator then.'
    try:
        with open(filepath, 'rb') as f:
            elf = parse_elf(f, interpreter=True, dynamic_section=True)
            return elf.has_pt_dynamic and (elf.has_soname or not elf.has_pt_interp)
    except (IOError, OSError, ElfParsingError):
        return False

class SharedLibrariesVisitor(BaseDirectoryVisitor):
    """Visitor that collects all shared libraries in a prefix, with the
    exception of an exclude list."""

    def __init__(self, exclude_list):
        if False:
            while True:
                i = 10
        self.exclude_list = frozenset(exclude_list)
        self.libraries = dict()
        self.excluded_through_symlink = set()

    def visit_file(self, root, rel_path, depth):
        if False:
            print('Hello World!')
        basename = os.path.basename(rel_path)
        if basename in self.exclude_list:
            return
        filepath = os.path.join(root, rel_path)
        s = os.lstat(filepath)
        identifier = (s.st_ino, s.st_dev)
        if identifier in self.libraries or identifier in self.excluded_through_symlink:
            return
        if is_shared_library_elf(filepath):
            self.libraries[identifier] = rel_path

    def visit_symlinked_file(self, root, rel_path, depth):
        if False:
            for i in range(10):
                print('nop')
        basename = os.path.basename(rel_path)
        if basename not in self.exclude_list:
            return
        filepath = os.path.join(root, rel_path)
        try:
            s = os.stat(filepath)
        except OSError:
            return
        self.excluded_through_symlink.add((s.st_ino, s.st_dev))

    def before_visit_dir(self, root, rel_path, depth):
        if False:
            print('Hello World!')
        return os.path.basename(rel_path) not in self.exclude_list

    def before_visit_symlinked_dir(self, root, rel_path, depth):
        if False:
            for i in range(10):
                print('nop')
        return False

    def get_shared_libraries_relative_paths(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the libraries that should be patched, with the excluded libraries\n        removed.'
        for identifier in self.excluded_through_symlink:
            self.libraries.pop(identifier, None)
        return [rel_path for rel_path in self.libraries.values()]

def patch_sonames(patchelf, root, rel_paths):
    if False:
        i = 10
        return i + 15
    "Set the soname to the file's own path for a list of\n    given shared libraries."
    fixed = []
    for rel_path in rel_paths:
        filepath = os.path.join(root, rel_path)
        normalized = os.path.normpath(filepath)
        args = ['--set-soname', normalized, normalized]
        output = patchelf(*args, output=str, error=str, fail_on_error=False)
        if patchelf.returncode == 0:
            fixed.append(rel_path)
        else:
            tty.warn('patchelf: failed to set soname of {}: {}'.format(normalized, output.strip()))
    return fixed

def find_and_patch_sonames(prefix, exclude_list, patchelf):
    if False:
        print('Hello World!')
    visitor = SharedLibrariesVisitor(exclude_list)
    visit_directory_tree(prefix, visitor)
    relative_paths = visitor.get_shared_libraries_relative_paths()
    return patch_sonames(patchelf, prefix, relative_paths)

def post_install(spec, explicit=None):
    if False:
        print('Hello World!')
    if not spack.config.get('config:shared_linking:bind', False):
        return
    if spec.external:
        return
    if not spec.satisfies('platform=linux') and (not spec.satisfies('platform=cray')):
        return
    if spack.bootstrap.is_bootstrapping():
        return
    patchelf_path = spack.relocate._patchelf()
    if not patchelf_path:
        return
    patchelf = Executable(patchelf_path)
    fixes = find_and_patch_sonames(spec.prefix, spec.package.non_bindable_shared_objects, patchelf)
    if not fixes:
        return
    tty.info('{}: Patched {} {}: {}'.format(spec.name, len(fixes), 'soname' if len(fixes) == 1 else 'sonames', ', '.join(elide_list(fixes, max_num=5))))