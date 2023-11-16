"""Adds docstrings to Storage functions"""
import torch._C
from torch._C import _add_docstr as add_docstr
storage_classes = ['StorageBase']

def add_docstr_all(method, docstr):
    if False:
        while True:
            i = 10
    for cls_name in storage_classes:
        cls = getattr(torch._C, cls_name)
        try:
            add_docstr(getattr(cls, method), docstr)
        except AttributeError:
            pass
add_docstr_all('from_file', '\nfrom_file(filename, shared=False, size=0) -> Storage\n\nCreates a CPU storage backed by a memory-mapped file.\n\nIf ``shared`` is ``True``, then memory is shared between all processes.\nAll changes are written to the file. If ``shared`` is ``False``, then the changes on\nthe storage do not affect the file.\n\n``size`` is the number of elements in the storage. If ``shared`` is ``False``,\nthen the file must contain at least ``size * sizeof(Type)`` bytes\n(``Type`` is the type of storage, in the case of an ``UnTypedStorage`` the file must contain at\nleast ``size`` bytes). If ``shared`` is ``True`` the file will be created if needed.\n\nArgs:\n    filename (str): file name to map\n    shared (bool): whether to share memory (whether ``MAP_SHARED`` or ``MAP_PRIVATE`` is passed to the\n                    underlying `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_)\n    size (int): number of elements in the storage\n')