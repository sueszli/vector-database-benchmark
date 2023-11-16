"""Inventory delta serialisation.

See doc/developers/inventory.txt for the description of the format.

In this module the interesting classes are:
 - InventoryDeltaSerializer - object to read/write inventory deltas.
"""
from __future__ import absolute_import
__all__ = ['InventoryDeltaSerializer']
from bzrlib import errors
from bzrlib.osutils import basename
from bzrlib import inventory
from bzrlib.revision import NULL_REVISION
FORMAT_1 = 'bzr inventory delta v1 (bzr 1.14)'

class InventoryDeltaError(errors.BzrError):
    """An error when serializing or deserializing an inventory delta."""
    internal_error = True

class IncompatibleInventoryDelta(errors.BzrError):
    """The delta could not be deserialised because its contents conflict with
    the allow_versioned_root or allow_tree_references flags of the
    deserializer.
    """
    internal_error = False

def _directory_content(entry):
    if False:
        while True:
            i = 10
    'Serialize the content component of entry which is a directory.\n    \n    :param entry: An InventoryDirectory.\n    '
    return 'dir'

def _file_content(entry):
    if False:
        print('Hello World!')
    'Serialize the content component of entry which is a file.\n    \n    :param entry: An InventoryFile.\n    '
    if entry.executable:
        exec_bytes = 'Y'
    else:
        exec_bytes = ''
    size_exec_sha = (entry.text_size, exec_bytes, entry.text_sha1)
    if None in size_exec_sha:
        raise InventoryDeltaError('Missing size or sha for %s' % entry.file_id)
    return 'file\x00%d\x00%s\x00%s' % size_exec_sha

def _link_content(entry):
    if False:
        print('Hello World!')
    'Serialize the content component of entry which is a symlink.\n    \n    :param entry: An InventoryLink.\n    '
    target = entry.symlink_target
    if target is None:
        raise InventoryDeltaError('Missing target for %s' % entry.file_id)
    return 'link\x00%s' % target.encode('utf8')

def _reference_content(entry):
    if False:
        for i in range(10):
            print('nop')
    'Serialize the content component of entry which is a tree-reference.\n    \n    :param entry: A TreeReference.\n    '
    tree_revision = entry.reference_revision
    if tree_revision is None:
        raise InventoryDeltaError('Missing reference revision for %s' % entry.file_id)
    return 'tree\x00%s' % tree_revision

def _dir_to_entry(content, name, parent_id, file_id, last_modified, _type=inventory.InventoryDirectory):
    if False:
        return 10
    'Convert a dir content record to an InventoryDirectory.'
    result = _type(file_id, name, parent_id)
    result.revision = last_modified
    return result

def _file_to_entry(content, name, parent_id, file_id, last_modified, _type=inventory.InventoryFile):
    if False:
        print('Hello World!')
    'Convert a dir content record to an InventoryFile.'
    result = _type(file_id, name, parent_id)
    result.revision = last_modified
    result.text_size = int(content[1])
    result.text_sha1 = content[3]
    if content[2]:
        result.executable = True
    else:
        result.executable = False
    return result

def _link_to_entry(content, name, parent_id, file_id, last_modified, _type=inventory.InventoryLink):
    if False:
        for i in range(10):
            print('nop')
    'Convert a link content record to an InventoryLink.'
    result = _type(file_id, name, parent_id)
    result.revision = last_modified
    result.symlink_target = content[1].decode('utf8')
    return result

def _tree_to_entry(content, name, parent_id, file_id, last_modified, _type=inventory.TreeReference):
    if False:
        for i in range(10):
            print('nop')
    'Convert a tree content record to a TreeReference.'
    result = _type(file_id, name, parent_id)
    result.revision = last_modified
    result.reference_revision = content[1]
    return result

class InventoryDeltaSerializer(object):
    """Serialize inventory deltas."""

    def __init__(self, versioned_root, tree_references):
        if False:
            for i in range(10):
                print('nop')
        'Create an InventoryDeltaSerializer.\n\n        :param versioned_root: If True, any root entry that is seen is expected\n            to be versioned, and root entries can have any fileid.\n        :param tree_references: If True support tree-reference entries.\n        '
        self._versioned_root = versioned_root
        self._tree_references = tree_references
        self._entry_to_content = {'directory': _directory_content, 'file': _file_content, 'symlink': _link_content}
        if tree_references:
            self._entry_to_content['tree-reference'] = _reference_content

    def delta_to_lines(self, old_name, new_name, delta_to_new):
        if False:
            i = 10
            return i + 15
        'Return a line sequence for delta_to_new.\n\n        Both the versioned_root and tree_references flags must be set via\n        require_flags before calling this.\n\n        :param old_name: A UTF8 revision id for the old inventory.  May be\n            NULL_REVISION if there is no older inventory and delta_to_new\n            includes the entire inventory contents.\n        :param new_name: The version name of the inventory we create with this\n            delta.\n        :param delta_to_new: An inventory delta such as Inventory.apply_delta\n            takes.\n        :return: The serialized delta as lines.\n        '
        if type(old_name) is not str:
            raise TypeError('old_name should be str, got %r' % (old_name,))
        if type(new_name) is not str:
            raise TypeError('new_name should be str, got %r' % (new_name,))
        lines = ['', '', '', '', '']
        to_line = self._delta_item_to_line
        for delta_item in delta_to_new:
            line = to_line(delta_item, new_name)
            if line.__class__ != str:
                raise InventoryDeltaError('to_line generated non-str output %r' % lines[-1])
            lines.append(line)
        lines.sort()
        lines[0] = 'format: %s\n' % FORMAT_1
        lines[1] = 'parent: %s\n' % old_name
        lines[2] = 'version: %s\n' % new_name
        lines[3] = 'versioned_root: %s\n' % self._serialize_bool(self._versioned_root)
        lines[4] = 'tree_references: %s\n' % self._serialize_bool(self._tree_references)
        return lines

    def _serialize_bool(self, value):
        if False:
            return 10
        if value:
            return 'true'
        else:
            return 'false'

    def _delta_item_to_line(self, delta_item, new_version):
        if False:
            return 10
        'Convert delta_item to a line.'
        (oldpath, newpath, file_id, entry) = delta_item
        if newpath is None:
            oldpath_utf8 = '/' + oldpath.encode('utf8')
            newpath_utf8 = 'None'
            parent_id = ''
            last_modified = NULL_REVISION
            content = 'deleted\x00\x00'
        else:
            if oldpath is None:
                oldpath_utf8 = 'None'
            else:
                oldpath_utf8 = '/' + oldpath.encode('utf8')
            if newpath == '/':
                raise AssertionError("Bad inventory delta: '/' is not a valid newpath (should be '') in delta item %r" % (delta_item,))
            newpath_utf8 = '/' + newpath.encode('utf8')
            parent_id = entry.parent_id or ''
            last_modified = entry.revision
            if newpath_utf8 == '/' and (not self._versioned_root):
                if last_modified != new_version:
                    raise InventoryDeltaError('Version present for / in %s (%s != %s)' % (file_id, last_modified, new_version))
            if last_modified is None:
                raise InventoryDeltaError('no version for fileid %s' % file_id)
            content = self._entry_to_content[entry.kind](entry)
        return '%s\x00%s\x00%s\x00%s\x00%s\x00%s\n' % (oldpath_utf8, newpath_utf8, file_id, parent_id, last_modified, content)

class InventoryDeltaDeserializer(object):
    """Deserialize inventory deltas."""

    def __init__(self, allow_versioned_root=True, allow_tree_references=True):
        if False:
            i = 10
            return i + 15
        'Create an InventoryDeltaDeserializer.\n\n        :param versioned_root: If True, any root entry that is seen is expected\n            to be versioned, and root entries can have any fileid.\n        :param tree_references: If True support tree-reference entries.\n        '
        self._allow_versioned_root = allow_versioned_root
        self._allow_tree_references = allow_tree_references

    def _deserialize_bool(self, value):
        if False:
            print('Hello World!')
        if value == 'true':
            return True
        elif value == 'false':
            return False
        else:
            raise InventoryDeltaError('value %r is not a bool' % (value,))

    def parse_text_bytes(self, bytes):
        if False:
            print('Hello World!')
        "Parse the text bytes of a serialized inventory delta.\n\n        If versioned_root and/or tree_references flags were set via\n        require_flags, then the parsed flags must match or a BzrError will be\n        raised.\n\n        :param bytes: The bytes to parse. This can be obtained by calling\n            delta_to_lines and then doing ''.join(delta_lines).\n        :return: (parent_id, new_id, versioned_root, tree_references,\n            inventory_delta)\n        "
        if bytes[-1:] != '\n':
            last_line = bytes.rsplit('\n', 1)[-1]
            raise InventoryDeltaError('last line not empty: %r' % (last_line,))
        lines = bytes.split('\n')[:-1]
        if not lines or lines[0] != 'format: %s' % FORMAT_1:
            raise InventoryDeltaError('unknown format %r' % lines[0:1])
        if len(lines) < 2 or not lines[1].startswith('parent: '):
            raise InventoryDeltaError('missing parent: marker')
        delta_parent_id = lines[1][8:]
        if len(lines) < 3 or not lines[2].startswith('version: '):
            raise InventoryDeltaError('missing version: marker')
        delta_version_id = lines[2][9:]
        if len(lines) < 4 or not lines[3].startswith('versioned_root: '):
            raise InventoryDeltaError('missing versioned_root: marker')
        delta_versioned_root = self._deserialize_bool(lines[3][16:])
        if len(lines) < 5 or not lines[4].startswith('tree_references: '):
            raise InventoryDeltaError('missing tree_references: marker')
        delta_tree_references = self._deserialize_bool(lines[4][17:])
        if not self._allow_versioned_root and delta_versioned_root:
            raise IncompatibleInventoryDelta('versioned_root not allowed')
        result = []
        seen_ids = set()
        line_iter = iter(lines)
        for i in range(5):
            line_iter.next()
        for line in line_iter:
            (oldpath_utf8, newpath_utf8, file_id, parent_id, last_modified, content) = line.split('\x00', 5)
            parent_id = parent_id or None
            if file_id in seen_ids:
                raise InventoryDeltaError('duplicate file id in inventory delta %r' % lines)
            seen_ids.add(file_id)
            if newpath_utf8 == '/' and (not delta_versioned_root) and (last_modified != delta_version_id):
                raise InventoryDeltaError('Versioned root found: %r' % line)
            elif newpath_utf8 != 'None' and last_modified[-1] == ':':
                raise InventoryDeltaError('special revisionid found: %r' % line)
            if content.startswith('tree\x00'):
                if delta_tree_references is False:
                    raise InventoryDeltaError('Tree reference found (but header said tree_references: false): %r' % line)
                elif not self._allow_tree_references:
                    raise IncompatibleInventoryDelta('Tree reference not allowed')
            if oldpath_utf8 == 'None':
                oldpath = None
            elif oldpath_utf8[:1] != '/':
                raise InventoryDeltaError('oldpath invalid (does not start with /): %r' % (oldpath_utf8,))
            else:
                oldpath_utf8 = oldpath_utf8[1:]
                oldpath = oldpath_utf8.decode('utf8')
            if newpath_utf8 == 'None':
                newpath = None
            elif newpath_utf8[:1] != '/':
                raise InventoryDeltaError('newpath invalid (does not start with /): %r' % (newpath_utf8,))
            else:
                newpath_utf8 = newpath_utf8[1:]
                newpath = newpath_utf8.decode('utf8')
            content_tuple = tuple(content.split('\x00'))
            if content_tuple[0] == 'deleted':
                entry = None
            else:
                entry = _parse_entry(newpath, file_id, parent_id, last_modified, content_tuple)
            delta_item = (oldpath, newpath, file_id, entry)
            result.append(delta_item)
        return (delta_parent_id, delta_version_id, delta_versioned_root, delta_tree_references, result)

def _parse_entry(path, file_id, parent_id, last_modified, content):
    if False:
        for i in range(10):
            print('nop')
    entry_factory = {'dir': _dir_to_entry, 'file': _file_to_entry, 'link': _link_to_entry, 'tree': _tree_to_entry}
    kind = content[0]
    if path.startswith('/'):
        raise AssertionError
    name = basename(path)
    return entry_factory[content[0]](content, name, parent_id, file_id, last_modified)