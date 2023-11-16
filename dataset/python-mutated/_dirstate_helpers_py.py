"""Python implementations of Dirstate Helper functions."""
from __future__ import absolute_import
import binascii
import os
import struct
from bzrlib import errors
from bzrlib.dirstate import DirState

def pack_stat(st, _b64=binascii.b2a_base64, _pack=struct.Struct('>6L').pack):
    if False:
        while True:
            i = 10
    'Convert stat values into a packed representation\n\n    Not all of the fields from the stat included are strictly needed, and by\n    just encoding the mtime and mode a slight speed increase could be gained.\n    However, using the pyrex version instead is a bigger win.\n    '
    return _b64(_pack(st.st_size & 4294967295, int(st.st_mtime) & 4294967295, int(st.st_ctime) & 4294967295, st.st_dev & 4294967295, st.st_ino & 4294967295, st.st_mode))[:-1]

def _unpack_stat(packed_stat):
    if False:
        return 10
    'Turn a packed_stat back into the stat fields.\n\n    This is meant as a debugging tool, should not be used in real code.\n    '
    (st_size, st_mtime, st_ctime, st_dev, st_ino, st_mode) = struct.unpack('>6L', binascii.a2b_base64(packed_stat))
    return dict(st_size=st_size, st_mtime=st_mtime, st_ctime=st_ctime, st_dev=st_dev, st_ino=st_ino, st_mode=st_mode)

def _bisect_path_left(paths, path):
    if False:
        while True:
            i = 10
    "Return the index where to insert path into paths.\n\n    This uses the dirblock sorting. So all children in a directory come before\n    the children of children. For example::\n\n        a/\n          b/\n            c\n          d/\n            e\n          b-c\n          d-e\n        a-a\n        a=c\n\n    Will be sorted as::\n\n        a\n        a-a\n        a=c\n        a/b\n        a/b-c\n        a/d\n        a/d-e\n        a/b/c\n        a/d/e\n\n    :param paths: A list of paths to search through\n    :param path: A single path to insert\n    :return: An offset where 'path' can be inserted.\n    :seealso: bisect.bisect_left\n    "
    hi = len(paths)
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        cur = paths[mid]
        if _cmp_path_by_dirblock(cur, path) < 0:
            lo = mid + 1
        else:
            hi = mid
    return lo

def _bisect_path_right(paths, path):
    if False:
        while True:
            i = 10
    "Return the index where to insert path into paths.\n\n    This uses a path-wise comparison so we get::\n        a\n        a-b\n        a=b\n        a/b\n    Rather than::\n        a\n        a-b\n        a/b\n        a=b\n    :param paths: A list of paths to search through\n    :param path: A single path to insert\n    :return: An offset where 'path' can be inserted.\n    :seealso: bisect.bisect_right\n    "
    hi = len(paths)
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        cur = paths[mid]
        if _cmp_path_by_dirblock(path, cur) < 0:
            hi = mid
        else:
            lo = mid + 1
    return lo

def bisect_dirblock(dirblocks, dirname, lo=0, hi=None, cache={}):
    if False:
        while True:
            i = 10
    'Return the index where to insert dirname into the dirblocks.\n\n    The return value idx is such that all directories blocks in dirblock[:idx]\n    have names < dirname, and all blocks in dirblock[idx:] have names >=\n    dirname.\n\n    Optional args lo (default 0) and hi (default len(dirblocks)) bound the\n    slice of a to be searched.\n    '
    if hi is None:
        hi = len(dirblocks)
    try:
        dirname_split = cache[dirname]
    except KeyError:
        dirname_split = dirname.split('/')
        cache[dirname] = dirname_split
    while lo < hi:
        mid = (lo + hi) // 2
        cur = dirblocks[mid][0]
        try:
            cur_split = cache[cur]
        except KeyError:
            cur_split = cur.split('/')
            cache[cur] = cur_split
        if cur_split < dirname_split:
            lo = mid + 1
        else:
            hi = mid
    return lo

def cmp_by_dirs(path1, path2):
    if False:
        print('Hello World!')
    'Compare two paths directory by directory.\n\n    This is equivalent to doing::\n\n       cmp(path1.split(\'/\'), path2.split(\'/\'))\n\n    The idea is that you should compare path components separately. This\n    differs from plain ``cmp(path1, path2)`` for paths like ``\'a-b\'`` and\n    ``a/b``. "a-b" comes after "a" but would come before "a/b" lexically.\n\n    :param path1: first path\n    :param path2: second path\n    :return: negative number if ``path1`` comes first,\n        0 if paths are equal,\n        and positive number if ``path2`` sorts first\n    '
    if not isinstance(path1, str):
        raise TypeError("'path1' must be a plain string, not %s: %r" % (type(path1), path1))
    if not isinstance(path2, str):
        raise TypeError("'path2' must be a plain string, not %s: %r" % (type(path2), path2))
    return cmp(path1.split('/'), path2.split('/'))

def _cmp_path_by_dirblock(path1, path2):
    if False:
        i = 10
        return i + 15
    'Compare two paths based on what directory they are in.\n\n    This generates a sort order, such that all children of a directory are\n    sorted together, and grandchildren are in the same order as the\n    children appear. But all grandchildren come after all children.\n\n    :param path1: first path\n    :param path2: the second path\n    :return: negative number if ``path1`` comes first,\n        0 if paths are equal\n        and a positive number if ``path2`` sorts first\n    '
    if not isinstance(path1, str):
        raise TypeError("'path1' must be a plain string, not %s: %r" % (type(path1), path1))
    if not isinstance(path2, str):
        raise TypeError("'path2' must be a plain string, not %s: %r" % (type(path2), path2))
    (dirname1, basename1) = os.path.split(path1)
    key1 = (dirname1.split('/'), basename1)
    (dirname2, basename2) = os.path.split(path2)
    key2 = (dirname2.split('/'), basename2)
    return cmp(key1, key2)

def _read_dirblocks(state):
    if False:
        return 10
    'Read in the dirblocks for the given DirState object.\n\n    This is tightly bound to the DirState internal representation. It should be\n    thought of as a member function, which is only separated out so that we can\n    re-write it in pyrex.\n\n    :param state: A DirState object.\n    :return: None\n    '
    state._state_file.seek(state._end_of_header)
    text = state._state_file.read()
    fields = text.split('\x00')
    trailing = fields.pop()
    if trailing != '':
        raise errors.DirstateCorrupt(state, 'trailing garbage: %r' % (trailing,))
    cur = 1
    num_present_parents = state._num_present_parents()
    tree_count = 1 + num_present_parents
    entry_size = state._fields_per_entry()
    expected_field_count = entry_size * state._num_entries
    field_count = len(fields)
    if field_count - cur != expected_field_count:
        raise errors.DirstateCorrupt(state, 'field count incorrect %s != %s, entry_size=%s, num_entries=%s fields=%r' % (field_count - cur, expected_field_count, entry_size, state._num_entries, fields))
    if num_present_parents == 1:
        _int = int
        next = iter(fields).next
        for x in xrange(cur):
            next()
        state._dirblocks = [('', []), ('', [])]
        current_block = state._dirblocks[0][1]
        current_dirname = ''
        append_entry = current_block.append
        for count in xrange(state._num_entries):
            dirname = next()
            name = next()
            file_id = next()
            if dirname != current_dirname:
                current_block = []
                current_dirname = dirname
                state._dirblocks.append((current_dirname, current_block))
                append_entry = current_block.append
            entry = ((current_dirname, name, file_id), [(next(), next(), _int(next()), next() == 'y', next()), (next(), next(), _int(next()), next() == 'y', next())])
            trailing = next()
            if trailing != '\n':
                raise ValueError('trailing garbage in dirstate: %r' % trailing)
            append_entry(entry)
        state._split_root_dirblock_into_contents()
    else:
        fields_to_entry = state._get_fields_to_entry()
        entries = [fields_to_entry(fields[pos:pos + entry_size]) for pos in xrange(cur, field_count, entry_size)]
        state._entries_to_current_state(entries)
    state._dirblock_state = DirState.IN_MEMORY_UNMODIFIED