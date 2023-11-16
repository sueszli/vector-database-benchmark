import os
from typing import IO, Optional, Tuple
import llnl.util.tty as tty
from llnl.util.filesystem import BaseDirectoryVisitor, visit_directory_tree
from spack.util.elf import ElfParsingError, parse_elf

def should_keep(path: bytes) -> bool:
    if False:
        i = 10
        return i + 15
    'Return True iff path starts with $ (typically for $ORIGIN/${ORIGIN}) or is\n    absolute and exists.'
    return path.startswith(b'$') or (os.path.isabs(path) and os.path.lexists(path))

def _drop_redundant_rpaths(f: IO) -> Optional[Tuple[bytes, bytes]]:
    if False:
        print('Hello World!')
    'Drop redundant entries from rpath.\n\n    Args:\n        f: File object to patch opened in r+b mode.\n\n    Returns:\n        A tuple of the old and new rpath if the rpath was patched, None otherwise.\n    '
    try:
        elf = parse_elf(f, interpreter=False, dynamic_section=True)
    except ElfParsingError:
        return None
    if not elf.has_rpath:
        return None
    old_rpath_str = elf.dt_rpath_str
    new_rpath_str = b':'.join((p for p in old_rpath_str.split(b':') if should_keep(p)))
    if old_rpath_str == new_rpath_str:
        return None
    pad = len(old_rpath_str) - len(new_rpath_str)
    if pad < 0:
        return None
    rpath_offset = elf.pt_dynamic_strtab_offset + elf.rpath_strtab_offset
    f.seek(rpath_offset)
    f.write(new_rpath_str + b'\x00' * pad)
    return (old_rpath_str, new_rpath_str)

def drop_redundant_rpaths(path: str) -> Optional[Tuple[bytes, bytes]]:
    if False:
        return 10
    'Drop redundant entries from rpath.\n\n    Args:\n        path: Path to a potential ELF file to patch.\n\n    Returns:\n        A tuple of the old and new rpath if the rpath was patched, None otherwise.\n    '
    try:
        with open(path, 'r+b') as f:
            return _drop_redundant_rpaths(f)
    except OSError:
        return None

class ElfFilesWithRPathVisitor(BaseDirectoryVisitor):
    """Visitor that collects all elf files that have an rpath"""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.visited = set()

    def visit_file(self, root, rel_path, depth):
        if False:
            i = 10
            return i + 15
        filepath = os.path.join(root, rel_path)
        s = os.lstat(filepath)
        identifier = (s.st_ino, s.st_dev)
        if s.st_nlink > 1:
            if identifier in self.visited:
                return
            self.visited.add(identifier)
        result = drop_redundant_rpaths(filepath)
        if result is not None:
            (old, new) = result
            tty.debug(f'Patched rpath in {rel_path} from {old!r} to {new!r}')

    def visit_symlinked_file(self, root, rel_path, depth):
        if False:
            i = 10
            return i + 15
        pass

    def before_visit_dir(self, root, rel_path, depth):
        if False:
            i = 10
            return i + 15
        return True

    def before_visit_symlinked_dir(self, root, rel_path, depth):
        if False:
            i = 10
            return i + 15
        return False

def post_install(spec, explicit=None):
    if False:
        for i in range(10):
            print('nop')
    if spec.external:
        return
    if not spec.satisfies('platform=linux') and (not spec.satisfies('platform=cray')):
        return
    visit_directory_tree(spec.prefix, ElfFilesWithRPathVisitor())