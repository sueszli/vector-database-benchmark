"""MemoryTree object.

See MemoryTree for more details.
"""
from __future__ import absolute_import
import os
from bzrlib import errors, mutabletree, revision as _mod_revision
from bzrlib.decorators import needs_read_lock
from bzrlib.inventory import Inventory
from bzrlib.osutils import sha_file
from bzrlib.mutabletree import needs_tree_write_lock
from bzrlib.transport.memory import MemoryTransport

class MemoryTree(mutabletree.MutableInventoryTree):
    """A MemoryTree is a specialisation of MutableTree.

    It maintains nearly no state outside of read_lock and write_lock
    transactions. (it keeps a reference to the branch, and its last-revision
    only).
    """

    def __init__(self, branch, revision_id):
        if False:
            for i in range(10):
                print('nop')
        'Construct a MemoryTree for branch using revision_id.'
        self.branch = branch
        self.bzrdir = branch.bzrdir
        self._branch_revision_id = revision_id
        self._locks = 0
        self._lock_mode = None

    def get_config_stack(self):
        if False:
            while True:
                i = 10
        return self.branch.get_config_stack()

    def is_control_filename(self, filename):
        if False:
            i = 10
            return i + 15
        return False

    @needs_tree_write_lock
    def _add(self, files, ids, kinds):
        if False:
            while True:
                i = 10
        'See MutableTree._add.'
        for (f, file_id, kind) in zip(files, ids, kinds):
            if kind is None:
                kind = 'file'
            if file_id is None:
                self._inventory.add_path(f, kind=kind)
            else:
                self._inventory.add_path(f, kind=kind, file_id=file_id)

    def basis_tree(self):
        if False:
            return 10
        'See Tree.basis_tree().'
        return self._basis_tree

    @staticmethod
    def create_on_branch(branch):
        if False:
            return 10
        'Create a MemoryTree for branch, using the last-revision of branch.'
        revision_id = _mod_revision.ensure_null(branch.last_revision())
        return MemoryTree(branch, revision_id)

    def _gather_kinds(self, files, kinds):
        if False:
            i = 10
            return i + 15
        'See MutableTree._gather_kinds.\n\n        This implementation does not care about the file kind of\n        missing files, so is a no-op.\n        '

    def get_file(self, file_id, path=None):
        if False:
            i = 10
            return i + 15
        'See Tree.get_file.'
        if path is None:
            path = self.id2path(file_id)
        return self._file_transport.get(path)

    def get_file_sha1(self, file_id, path=None, stat_value=None):
        if False:
            for i in range(10):
                print('nop')
        'See Tree.get_file_sha1().'
        if path is None:
            path = self.id2path(file_id)
        stream = self._file_transport.get(path)
        return sha_file(stream)

    def get_root_id(self):
        if False:
            while True:
                i = 10
        return self.path2id('')

    def _comparison_data(self, entry, path):
        if False:
            print('Hello World!')
        'See Tree._comparison_data.'
        if entry is None:
            return (None, False, None)
        return (entry.kind, entry.executable, None)

    @needs_tree_write_lock
    def rename_one(self, from_rel, to_rel):
        if False:
            for i in range(10):
                print('nop')
        file_id = self.path2id(from_rel)
        (to_dir, to_tail) = os.path.split(to_rel)
        to_parent_id = self.path2id(to_dir)
        self._file_transport.move(from_rel, to_rel)
        self._inventory.rename(file_id, to_parent_id, to_tail)

    def path_content_summary(self, path):
        if False:
            for i in range(10):
                print('nop')
        'See Tree.path_content_summary.'
        id = self.path2id(path)
        if id is None:
            return ('missing', None, None, None)
        kind = self.kind(id)
        if kind == 'file':
            bytes = self._file_transport.get_bytes(path)
            size = len(bytes)
            executable = self._inventory[id].executable
            sha1 = None
            return (kind, size, executable, sha1)
        elif kind == 'directory':
            return (kind, None, None, None)
        elif kind == 'symlink':
            raise NotImplementedError('symlink support')
        else:
            raise NotImplementedError('unknown kind')

    def _file_size(self, entry, stat_value):
        if False:
            i = 10
            return i + 15
        'See Tree._file_size.'
        if entry is None:
            return 0
        return entry.text_size

    @needs_read_lock
    def get_parent_ids(self):
        if False:
            i = 10
            return i + 15
        'See Tree.get_parent_ids.\n\n        This implementation returns the current cached value from\n            self._parent_ids.\n        '
        return list(self._parent_ids)

    def has_filename(self, filename):
        if False:
            i = 10
            return i + 15
        'See Tree.has_filename().'
        return self._file_transport.has(filename)

    def is_executable(self, file_id, path=None):
        if False:
            i = 10
            return i + 15
        return self._inventory[file_id].executable

    def kind(self, file_id):
        if False:
            print('Hello World!')
        return self._inventory[file_id].kind

    def mkdir(self, path, file_id=None):
        if False:
            for i in range(10):
                print('nop')
        'See MutableTree.mkdir().'
        self.add(path, file_id, 'directory')
        if file_id is None:
            file_id = self.path2id(path)
        self._file_transport.mkdir(path)
        return file_id

    @needs_read_lock
    def last_revision(self):
        if False:
            while True:
                i = 10
        'See MutableTree.last_revision.'
        return self._branch_revision_id

    def lock_read(self):
        if False:
            while True:
                i = 10
        'Lock the memory tree for reading.\n\n        This triggers population of data from the branch for its revision.\n        '
        self._locks += 1
        try:
            if self._locks == 1:
                self.branch.lock_read()
                self._lock_mode = 'r'
                self._populate_from_branch()
        except:
            self._locks -= 1
            raise

    def lock_tree_write(self):
        if False:
            for i in range(10):
                print('nop')
        'See MutableTree.lock_tree_write().'
        self._locks += 1
        try:
            if self._locks == 1:
                self.branch.lock_read()
                self._lock_mode = 'w'
                self._populate_from_branch()
            elif self._lock_mode == 'r':
                raise errors.ReadOnlyError(self)
        except:
            self._locks -= 1
            raise

    def lock_write(self):
        if False:
            for i in range(10):
                print('nop')
        'See MutableTree.lock_write().'
        self._locks += 1
        try:
            if self._locks == 1:
                self.branch.lock_write()
                self._lock_mode = 'w'
                self._populate_from_branch()
            elif self._lock_mode == 'r':
                raise errors.ReadOnlyError(self)
        except:
            self._locks -= 1
            raise

    def _populate_from_branch(self):
        if False:
            return 10
        'Populate the in-tree state from the branch.'
        self._set_basis()
        if self._branch_revision_id == _mod_revision.NULL_REVISION:
            self._parent_ids = []
        else:
            self._parent_ids = [self._branch_revision_id]
        self._inventory = Inventory(None, self._basis_tree.get_revision_id())
        self._file_transport = MemoryTransport()
        inventory_entries = self._basis_tree.iter_entries_by_dir()
        for (path, entry) in inventory_entries:
            self._inventory.add(entry.copy())
            if path == '':
                continue
            if entry.kind == 'directory':
                self._file_transport.mkdir(path)
            elif entry.kind == 'file':
                self._file_transport.put_file(path, self._basis_tree.get_file(entry.file_id))
            else:
                raise NotImplementedError(self._populate_from_branch)

    def put_file_bytes_non_atomic(self, file_id, bytes):
        if False:
            print('Hello World!')
        'See MutableTree.put_file_bytes_non_atomic.'
        self._file_transport.put_bytes(self.id2path(file_id), bytes)

    def unlock(self):
        if False:
            while True:
                i = 10
        'Release a lock.\n\n        This frees all cached state when the last lock context for the tree is\n        left.\n        '
        if self._locks == 1:
            self._basis_tree = None
            self._parent_ids = []
            self._inventory = None
            try:
                self.branch.unlock()
            finally:
                self._locks = 0
                self._lock_mode = None
        else:
            self._locks -= 1

    @needs_tree_write_lock
    def unversion(self, file_ids):
        if False:
            while True:
                i = 10
        'Remove the file ids in file_ids from the current versioned set.\n\n        When a file_id is unversioned, all of its children are automatically\n        unversioned.\n\n        :param file_ids: The file ids to stop versioning.\n        :raises: NoSuchId if any fileid is not currently versioned.\n        '
        for file_id in file_ids:
            if self._inventory.has_id(file_id):
                self._inventory.remove_recursive_id(file_id)
            else:
                raise errors.NoSuchId(self, file_id)

    def set_parent_ids(self, revision_ids, allow_leftmost_as_ghost=False):
        if False:
            return 10
        'See MutableTree.set_parent_trees().'
        for revision_id in revision_ids:
            _mod_revision.check_not_reserved_id(revision_id)
        if len(revision_ids) == 0:
            self._parent_ids = []
            self._branch_revision_id = _mod_revision.NULL_REVISION
        else:
            self._parent_ids = revision_ids
            self._branch_revision_id = revision_ids[0]
        self._allow_leftmost_as_ghost = allow_leftmost_as_ghost
        self._set_basis()

    def _set_basis(self):
        if False:
            return 10
        try:
            self._basis_tree = self.branch.repository.revision_tree(self._branch_revision_id)
        except errors.NoSuchRevision:
            if self._allow_leftmost_as_ghost:
                self._basis_tree = self.branch.repository.revision_tree(_mod_revision.NULL_REVISION)
            else:
                raise

    def set_parent_trees(self, parents_list, allow_leftmost_as_ghost=False):
        if False:
            print('Hello World!')
        'See MutableTree.set_parent_trees().'
        if len(parents_list) == 0:
            self._parent_ids = []
            self._basis_tree = self.branch.repository.revision_tree(_mod_revision.NULL_REVISION)
        else:
            if parents_list[0][1] is None and (not allow_leftmost_as_ghost):
                raise errors.GhostRevisionUnusableHere(parents_list[0][0])
            self._parent_ids = [parent_id for (parent_id, tree) in parents_list]
            if parents_list[0][1] is None or parents_list[0][1] == 'null:':
                self._basis_tree = self.branch.repository.revision_tree(_mod_revision.NULL_REVISION)
            else:
                self._basis_tree = parents_list[0][1]
            self._branch_revision_id = parents_list[0][0]