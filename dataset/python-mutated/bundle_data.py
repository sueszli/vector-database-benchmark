"""Read in a bundle stream, and process it into a BundleReader object."""
from __future__ import absolute_import
import base64
from cStringIO import StringIO
import os
import pprint
from bzrlib import osutils, timestamp
from bzrlib.bundle import apply_bundle
from bzrlib.errors import TestamentMismatch, BzrError
from bzrlib.inventory import Inventory, InventoryDirectory, InventoryFile, InventoryLink
from bzrlib.osutils import sha_string, pathjoin
from bzrlib.revision import Revision, NULL_REVISION
from bzrlib.testament import StrictTestament
from bzrlib.trace import mutter, warning
from bzrlib.tree import Tree
from bzrlib.xml5 import serializer_v5

class RevisionInfo(object):
    """Gets filled out for each revision object that is read.
    """

    def __init__(self, revision_id):
        if False:
            i = 10
            return i + 15
        self.revision_id = revision_id
        self.sha1 = None
        self.committer = None
        self.date = None
        self.timestamp = None
        self.timezone = None
        self.inventory_sha1 = None
        self.parent_ids = None
        self.base_id = None
        self.message = None
        self.properties = None
        self.tree_actions = None

    def __str__(self):
        if False:
            return 10
        return pprint.pformat(self.__dict__)

    def as_revision(self):
        if False:
            print('Hello World!')
        rev = Revision(revision_id=self.revision_id, committer=self.committer, timestamp=float(self.timestamp), timezone=int(self.timezone), inventory_sha1=self.inventory_sha1, message='\n'.join(self.message))
        if self.parent_ids:
            rev.parent_ids.extend(self.parent_ids)
        if self.properties:
            for property in self.properties:
                key_end = property.find(': ')
                if key_end == -1:
                    if not property.endswith(':'):
                        raise ValueError(property)
                    key = str(property[:-1])
                    value = ''
                else:
                    key = str(property[:key_end])
                    value = property[key_end + 2:]
                rev.properties[key] = value
        return rev

    @staticmethod
    def from_revision(revision):
        if False:
            print('Hello World!')
        revision_info = RevisionInfo(revision.revision_id)
        date = timestamp.format_highres_date(revision.timestamp, revision.timezone)
        revision_info.date = date
        revision_info.timezone = revision.timezone
        revision_info.timestamp = revision.timestamp
        revision_info.message = revision.message.split('\n')
        revision_info.properties = [': '.join(p) for p in revision.properties.iteritems()]
        return revision_info

class BundleInfo(object):
    """This contains the meta information. Stuff that allows you to
    recreate the revision or inventory XML.
    """

    def __init__(self, bundle_format=None):
        if False:
            i = 10
            return i + 15
        self.bundle_format = None
        self.committer = None
        self.date = None
        self.message = None
        self.revisions = []
        self.real_revisions = []
        self.timestamp = None
        self.timezone = None
        self._validated_revisions_against_repo = False

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return pprint.pformat(self.__dict__)

    def complete_info(self):
        if False:
            return 10
        'This makes sure that all information is properly\n        split up, based on the assumptions that can be made\n        when information is missing.\n        '
        from bzrlib.timestamp import unpack_highres_date
        if not self.timestamp and self.date:
            (self.timestamp, self.timezone) = unpack_highres_date(self.date)
        self.real_revisions = []
        for rev in self.revisions:
            if rev.timestamp is None:
                if rev.date is not None:
                    (rev.timestamp, rev.timezone) = unpack_highres_date(rev.date)
                else:
                    rev.timestamp = self.timestamp
                    rev.timezone = self.timezone
            if rev.message is None and self.message:
                rev.message = self.message
            if rev.committer is None and self.committer:
                rev.committer = self.committer
            self.real_revisions.append(rev.as_revision())

    def get_base(self, revision):
        if False:
            while True:
                i = 10
        revision_info = self.get_revision_info(revision.revision_id)
        if revision_info.base_id is not None:
            return revision_info.base_id
        if len(revision.parent_ids) == 0:
            return NULL_REVISION
        else:
            return revision.parent_ids[-1]

    def _get_target(self):
        if False:
            while True:
                i = 10
        'Return the target revision.'
        if len(self.real_revisions) > 0:
            return self.real_revisions[0].revision_id
        elif len(self.revisions) > 0:
            return self.revisions[0].revision_id
        return None
    target = property(_get_target, doc='The target revision id')

    def get_revision(self, revision_id):
        if False:
            i = 10
            return i + 15
        for r in self.real_revisions:
            if r.revision_id == revision_id:
                return r
        raise KeyError(revision_id)

    def get_revision_info(self, revision_id):
        if False:
            i = 10
            return i + 15
        for r in self.revisions:
            if r.revision_id == revision_id:
                return r
        raise KeyError(revision_id)

    def revision_tree(self, repository, revision_id, base=None):
        if False:
            return 10
        revision = self.get_revision(revision_id)
        base = self.get_base(revision)
        if base == revision_id:
            raise AssertionError()
        if not self._validated_revisions_against_repo:
            self._validate_references_from_repository(repository)
        revision_info = self.get_revision_info(revision_id)
        inventory_revision_id = revision_id
        bundle_tree = BundleTree(repository.revision_tree(base), inventory_revision_id)
        self._update_tree(bundle_tree, revision_id)
        inv = bundle_tree.inventory
        self._validate_inventory(inv, revision_id)
        self._validate_revision(bundle_tree, revision_id)
        return bundle_tree

    def _validate_references_from_repository(self, repository):
        if False:
            for i in range(10):
                print('nop')
        'Now that we have a repository which should have some of the\n        revisions we care about, go through and validate all of them\n        that we can.\n        '
        rev_to_sha = {}
        inv_to_sha = {}

        def add_sha(d, revision_id, sha1):
            if False:
                for i in range(10):
                    print('nop')
            if revision_id is None:
                if sha1 is not None:
                    raise BzrError('A Null revision should alwayshave a null sha1 hash')
                return
            if revision_id in d:
                if sha1 != d[revision_id]:
                    raise BzrError('** Revision %r referenced with 2 different sha hashes %s != %s' % (revision_id, sha1, d[revision_id]))
            else:
                d[revision_id] = sha1
        checked = {}
        for rev_info in self.revisions:
            checked[rev_info.revision_id] = True
            add_sha(rev_to_sha, rev_info.revision_id, rev_info.sha1)
        for (rev, rev_info) in zip(self.real_revisions, self.revisions):
            add_sha(inv_to_sha, rev_info.revision_id, rev_info.inventory_sha1)
        count = 0
        missing = {}
        for (revision_id, sha1) in rev_to_sha.iteritems():
            if repository.has_revision(revision_id):
                testament = StrictTestament.from_revision(repository, revision_id)
                local_sha1 = self._testament_sha1_from_revision(repository, revision_id)
                if sha1 != local_sha1:
                    raise BzrError('sha1 mismatch. For revision id {%s}local: %s, bundle: %s' % (revision_id, local_sha1, sha1))
                else:
                    count += 1
            elif revision_id not in checked:
                missing[revision_id] = sha1
        if len(missing) > 0:
            warning('Not all revision hashes could be validated. Unable validate %d hashes' % len(missing))
        mutter('Verified %d sha hashes for the bundle.' % count)
        self._validated_revisions_against_repo = True

    def _validate_inventory(self, inv, revision_id):
        if False:
            i = 10
            return i + 15
        'At this point we should have generated the BundleTree,\n        so build up an inventory, and make sure the hashes match.\n        '
        s = serializer_v5.write_inventory_to_string(inv)
        sha1 = sha_string(s)
        rev = self.get_revision(revision_id)
        if rev.revision_id != revision_id:
            raise AssertionError()
        if sha1 != rev.inventory_sha1:
            f = open(',,bogus-inv', 'wb')
            try:
                f.write(s)
            finally:
                f.close()
            warning('Inventory sha hash mismatch for revision %s. %s != %s' % (revision_id, sha1, rev.inventory_sha1))

    def _validate_revision(self, tree, revision_id):
        if False:
            print('Hello World!')
        'Make sure all revision entries match their checksum.'
        rev_to_sha1 = {}
        rev = self.get_revision(revision_id)
        rev_info = self.get_revision_info(revision_id)
        if not rev.revision_id == rev_info.revision_id:
            raise AssertionError()
        if not rev.revision_id == revision_id:
            raise AssertionError()
        sha1 = self._testament_sha1(rev, tree)
        if sha1 != rev_info.sha1:
            raise TestamentMismatch(rev.revision_id, rev_info.sha1, sha1)
        if rev.revision_id in rev_to_sha1:
            raise BzrError('Revision {%s} given twice in the list' % rev.revision_id)
        rev_to_sha1[rev.revision_id] = sha1

    def _update_tree(self, bundle_tree, revision_id):
        if False:
            return 10
        'This fills out a BundleTree based on the information\n        that was read in.\n\n        :param bundle_tree: A BundleTree to update with the new information.\n        '

        def get_rev_id(last_changed, path, kind):
            if False:
                i = 10
                return i + 15
            if last_changed is not None:
                changed_revision_id = osutils.safe_revision_id(last_changed, warn=False)
            else:
                changed_revision_id = revision_id
            bundle_tree.note_last_changed(path, changed_revision_id)
            return changed_revision_id

        def extra_info(info, new_path):
            if False:
                print('Hello World!')
            last_changed = None
            encoding = None
            for info_item in info:
                try:
                    (name, value) = info_item.split(':', 1)
                except ValueError:
                    raise ValueError('Value %r has no colon' % info_item)
                if name == 'last-changed':
                    last_changed = value
                elif name == 'executable':
                    val = value == 'yes'
                    bundle_tree.note_executable(new_path, val)
                elif name == 'target':
                    bundle_tree.note_target(new_path, value)
                elif name == 'encoding':
                    encoding = value
            return (last_changed, encoding)

        def do_patch(path, lines, encoding):
            if False:
                print('Hello World!')
            if encoding == 'base64':
                patch = base64.decodestring(''.join(lines))
            elif encoding is None:
                patch = ''.join(lines)
            else:
                raise ValueError(encoding)
            bundle_tree.note_patch(path, patch)

        def renamed(kind, extra, lines):
            if False:
                for i in range(10):
                    print('nop')
            info = extra.split(' // ')
            if len(info) < 2:
                raise BzrError('renamed action lines need both a from and to: %r' % extra)
            old_path = info[0]
            if info[1].startswith('=> '):
                new_path = info[1][3:]
            else:
                new_path = info[1]
            bundle_tree.note_rename(old_path, new_path)
            (last_modified, encoding) = extra_info(info[2:], new_path)
            revision = get_rev_id(last_modified, new_path, kind)
            if lines:
                do_patch(new_path, lines, encoding)

        def removed(kind, extra, lines):
            if False:
                while True:
                    i = 10
            info = extra.split(' // ')
            if len(info) > 1:
                raise BzrError('removed action lines should only have the path: %r' % extra)
            path = info[0]
            bundle_tree.note_deletion(path)

        def added(kind, extra, lines):
            if False:
                print('Hello World!')
            info = extra.split(' // ')
            if len(info) <= 1:
                raise BzrError('add action lines require the path and file id: %r' % extra)
            elif len(info) > 5:
                raise BzrError('add action lines have fewer than 5 entries.: %r' % extra)
            path = info[0]
            if not info[1].startswith('file-id:'):
                raise BzrError('The file-id should follow the path for an add: %r' % extra)
            file_id = osutils.safe_file_id(info[1][8:], warn=False)
            bundle_tree.note_id(file_id, path, kind)
            bundle_tree.note_executable(path, False)
            (last_changed, encoding) = extra_info(info[2:], path)
            revision = get_rev_id(last_changed, path, kind)
            if kind == 'directory':
                return
            do_patch(path, lines, encoding)

        def modified(kind, extra, lines):
            if False:
                while True:
                    i = 10
            info = extra.split(' // ')
            if len(info) < 1:
                raise BzrError('modified action lines have at leastthe path in them: %r' % extra)
            path = info[0]
            (last_modified, encoding) = extra_info(info[1:], path)
            revision = get_rev_id(last_modified, path, kind)
            if lines:
                do_patch(path, lines, encoding)
        valid_actions = {'renamed': renamed, 'removed': removed, 'added': added, 'modified': modified}
        for (action_line, lines) in self.get_revision_info(revision_id).tree_actions:
            first = action_line.find(' ')
            if first == -1:
                raise BzrError('Bogus action line (no opening space): %r' % action_line)
            second = action_line.find(' ', first + 1)
            if second == -1:
                raise BzrError('Bogus action line (missing second space): %r' % action_line)
            action = action_line[:first]
            kind = action_line[first + 1:second]
            if kind not in ('file', 'directory', 'symlink'):
                raise BzrError('Bogus action line (invalid object kind %r): %r' % (kind, action_line))
            extra = action_line[second + 1:]
            if action not in valid_actions:
                raise BzrError('Bogus action line (unrecognized action): %r' % action_line)
            valid_actions[action](kind, extra, lines)

    def install_revisions(self, target_repo, stream_input=True):
        if False:
            return 10
        'Install revisions and return the target revision\n\n        :param target_repo: The repository to install into\n        :param stream_input: Ignored by this implementation.\n        '
        apply_bundle.install_bundle(target_repo, self)
        return self.target

    def get_merge_request(self, target_repo):
        if False:
            for i in range(10):
                print('nop')
        'Provide data for performing a merge\n\n        Returns suggested base, suggested target, and patch verification status\n        '
        return (None, self.target, 'inapplicable')

class BundleTree(Tree):

    def __init__(self, base_tree, revision_id):
        if False:
            i = 10
            return i + 15
        self.base_tree = base_tree
        self._renamed = {}
        self._renamed_r = {}
        self._new_id = {}
        self._new_id_r = {}
        self._kinds = {}
        self._last_changed = {}
        self._executable = {}
        self.patches = {}
        self._targets = {}
        self.deleted = []
        self.contents_by_id = True
        self.revision_id = revision_id
        self._inventory = None

    def __str__(self):
        if False:
            print('Hello World!')
        return pprint.pformat(self.__dict__)

    def note_rename(self, old_path, new_path):
        if False:
            print('Hello World!')
        'A file/directory has been renamed from old_path => new_path'
        if new_path in self._renamed:
            raise AssertionError(new_path)
        if old_path in self._renamed_r:
            raise AssertionError(old_path)
        self._renamed[new_path] = old_path
        self._renamed_r[old_path] = new_path

    def note_id(self, new_id, new_path, kind='file'):
        if False:
            while True:
                i = 10
        "Files that don't exist in base need a new id."
        self._new_id[new_path] = new_id
        self._new_id_r[new_id] = new_path
        self._kinds[new_id] = kind

    def note_last_changed(self, file_id, revision_id):
        if False:
            return 10
        if file_id in self._last_changed and self._last_changed[file_id] != revision_id:
            raise BzrError('Mismatched last-changed revision for file_id {%s}: %s != %s' % (file_id, self._last_changed[file_id], revision_id))
        self._last_changed[file_id] = revision_id

    def note_patch(self, new_path, patch):
        if False:
            while True:
                i = 10
        'There is a patch for a given filename.'
        self.patches[new_path] = patch

    def note_target(self, new_path, target):
        if False:
            print('Hello World!')
        'The symlink at the new path has the given target'
        self._targets[new_path] = target

    def note_deletion(self, old_path):
        if False:
            print('Hello World!')
        'The file at old_path has been deleted.'
        self.deleted.append(old_path)

    def note_executable(self, new_path, executable):
        if False:
            return 10
        self._executable[new_path] = executable

    def old_path(self, new_path):
        if False:
            while True:
                i = 10
        'Get the old_path (path in the base_tree) for the file at new_path'
        if new_path[:1] in ('\\', '/'):
            raise ValueError(new_path)
        old_path = self._renamed.get(new_path)
        if old_path is not None:
            return old_path
        (dirname, basename) = os.path.split(new_path)
        if dirname != '':
            old_dir = self.old_path(dirname)
            if old_dir is None:
                old_path = None
            else:
                old_path = pathjoin(old_dir, basename)
        else:
            old_path = new_path
        if old_path in self._renamed_r:
            return None
        return old_path

    def new_path(self, old_path):
        if False:
            return 10
        'Get the new_path (path in the target_tree) for the file at old_path\n        in the base tree.\n        '
        if old_path[:1] in ('\\', '/'):
            raise ValueError(old_path)
        new_path = self._renamed_r.get(old_path)
        if new_path is not None:
            return new_path
        if new_path in self._renamed:
            return None
        (dirname, basename) = os.path.split(old_path)
        if dirname != '':
            new_dir = self.new_path(dirname)
            if new_dir is None:
                new_path = None
            else:
                new_path = pathjoin(new_dir, basename)
        else:
            new_path = old_path
        if new_path in self._renamed:
            return None
        return new_path

    def get_root_id(self):
        if False:
            print('Hello World!')
        return self.path2id('')

    def path2id(self, path):
        if False:
            i = 10
            return i + 15
        'Return the id of the file present at path in the target tree.'
        file_id = self._new_id.get(path)
        if file_id is not None:
            return file_id
        old_path = self.old_path(path)
        if old_path is None:
            return None
        if old_path in self.deleted:
            return None
        return self.base_tree.path2id(old_path)

    def id2path(self, file_id):
        if False:
            return 10
        'Return the new path in the target tree of the file with id file_id'
        path = self._new_id_r.get(file_id)
        if path is not None:
            return path
        old_path = self.base_tree.id2path(file_id)
        if old_path is None:
            return None
        if old_path in self.deleted:
            return None
        return self.new_path(old_path)

    def old_contents_id(self, file_id):
        if False:
            while True:
                i = 10
        'Return the id in the base_tree for the given file_id.\n        Return None if the file did not exist in base.\n        '
        if self.contents_by_id:
            if self.base_tree.has_id(file_id):
                return file_id
            else:
                return None
        new_path = self.id2path(file_id)
        return self.base_tree.path2id(new_path)

    def get_file(self, file_id):
        if False:
            while True:
                i = 10
        'Return a file-like object containing the new contents of the\n        file given by file_id.\n\n        TODO:   It might be nice if this actually generated an entry\n                in the text-store, so that the file contents would\n                then be cached.\n        '
        base_id = self.old_contents_id(file_id)
        if base_id is not None and base_id != self.base_tree.get_root_id():
            patch_original = self.base_tree.get_file(base_id)
        else:
            patch_original = None
        file_patch = self.patches.get(self.id2path(file_id))
        if file_patch is None:
            if patch_original is None and self.kind(file_id) == 'directory':
                return StringIO()
            if patch_original is None:
                raise AssertionError('None: %s' % file_id)
            return patch_original
        if file_patch.startswith('\\'):
            raise ValueError('Malformed patch for %s, %r' % (file_id, file_patch))
        return patched_file(file_patch, patch_original)

    def get_symlink_target(self, file_id, path=None):
        if False:
            while True:
                i = 10
        if path is None:
            path = self.id2path(file_id)
        try:
            return self._targets[path]
        except KeyError:
            return self.base_tree.get_symlink_target(file_id)

    def kind(self, file_id):
        if False:
            print('Hello World!')
        if file_id in self._kinds:
            return self._kinds[file_id]
        return self.base_tree.kind(file_id)

    def get_file_revision(self, file_id):
        if False:
            for i in range(10):
                print('nop')
        path = self.id2path(file_id)
        if path in self._last_changed:
            return self._last_changed[path]
        else:
            return self.base_tree.get_file_revision(file_id)

    def is_executable(self, file_id):
        if False:
            while True:
                i = 10
        path = self.id2path(file_id)
        if path in self._executable:
            return self._executable[path]
        else:
            return self.base_tree.is_executable(file_id)

    def get_last_changed(self, file_id):
        if False:
            return 10
        path = self.id2path(file_id)
        if path in self._last_changed:
            return self._last_changed[path]
        return self.base_tree.get_file_revision(file_id)

    def get_size_and_sha1(self, file_id):
        if False:
            print('Hello World!')
        'Return the size and sha1 hash of the given file id.\n        If the file was not locally modified, this is extracted\n        from the base_tree. Rather than re-reading the file.\n        '
        new_path = self.id2path(file_id)
        if new_path is None:
            return (None, None)
        if new_path not in self.patches:
            text_size = self.base_tree.get_file_size(file_id)
            text_sha1 = self.base_tree.get_file_sha1(file_id)
            return (text_size, text_sha1)
        fileobj = self.get_file(file_id)
        content = fileobj.read()
        return (len(content), sha_string(content))

    def _get_inventory(self):
        if False:
            print('Hello World!')
        'Build up the inventory entry for the BundleTree.\n\n        This need to be called before ever accessing self.inventory\n        '
        from os.path import dirname, basename
        inv = Inventory(None, self.revision_id)

        def add_entry(file_id):
            if False:
                i = 10
                return i + 15
            path = self.id2path(file_id)
            if path is None:
                return
            if path == '':
                parent_id = None
            else:
                parent_path = dirname(path)
                parent_id = self.path2id(parent_path)
            kind = self.kind(file_id)
            revision_id = self.get_last_changed(file_id)
            name = basename(path)
            if kind == 'directory':
                ie = InventoryDirectory(file_id, name, parent_id)
            elif kind == 'file':
                ie = InventoryFile(file_id, name, parent_id)
                ie.executable = self.is_executable(file_id)
            elif kind == 'symlink':
                ie = InventoryLink(file_id, name, parent_id)
                ie.symlink_target = self.get_symlink_target(file_id, path)
            ie.revision = revision_id
            if kind == 'file':
                (ie.text_size, ie.text_sha1) = self.get_size_and_sha1(file_id)
                if ie.text_size is None:
                    raise BzrError('Got a text_size of None for file_id %r' % file_id)
            inv.add(ie)
        sorted_entries = self.sorted_path_id()
        for (path, file_id) in sorted_entries:
            add_entry(file_id)
        return inv
    inventory = property(_get_inventory)
    root_inventory = property(_get_inventory)

    def all_file_ids(self):
        if False:
            return 10
        return set([entry.file_id for (path, entry) in self.inventory.iter_entries()])

    def list_files(self, include_root=False, from_dir=None, recursive=True):
        if False:
            return 10
        inv = self.inventory
        if from_dir is None:
            from_dir_id = None
        else:
            from_dir_id = inv.path2id(from_dir)
            if from_dir_id is None:
                return
        entries = inv.iter_entries(from_dir=from_dir_id, recursive=recursive)
        if inv.root is not None and (not include_root) and (from_dir is None):
            entries.next()
        for (path, entry) in entries:
            yield (path, 'V', entry.kind, entry.file_id, entry)

    def sorted_path_id(self):
        if False:
            return 10
        paths = []
        for result in self._new_id.iteritems():
            paths.append(result)
        for id in self.base_tree.all_file_ids():
            path = self.id2path(id)
            if path is None:
                continue
            paths.append((path, id))
        paths.sort()
        return paths

def patched_file(file_patch, original):
    if False:
        print('Hello World!')
    'Produce a file-like object with the patched version of a text'
    from bzrlib.patches import iter_patched
    from bzrlib.iterablefile import IterableFile
    if file_patch == '':
        return IterableFile(())
    return IterableFile(iter_patched(original, StringIO(file_patch).readlines()))