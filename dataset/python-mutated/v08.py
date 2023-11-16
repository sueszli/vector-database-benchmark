"""Serializer factory for reading and writing bundles.
"""
from __future__ import absolute_import
from bzrlib import errors, ui
from bzrlib.bundle.serializer import BundleSerializer, _get_bundle_header
from bzrlib.bundle.serializer import binary_diff
from bzrlib.bundle.bundle_data import RevisionInfo, BundleInfo
from bzrlib.diff import internal_diff
from bzrlib.revision import NULL_REVISION
from bzrlib.testament import StrictTestament
from bzrlib.timestamp import format_highres_date
from bzrlib.textfile import text_file
from bzrlib.trace import mutter
bool_text = {True: 'yes', False: 'no'}

class Action(object):
    """Represent an action"""

    def __init__(self, name, parameters=None, properties=None):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        if parameters is None:
            self.parameters = []
        else:
            self.parameters = parameters
        if properties is None:
            self.properties = []
        else:
            self.properties = properties

    def add_utf8_property(self, name, value):
        if False:
            i = 10
            return i + 15
        'Add a property whose value is currently utf8 to the action.'
        self.properties.append((name, value.decode('utf8')))

    def add_property(self, name, value):
        if False:
            while True:
                i = 10
        'Add a property to the action'
        self.properties.append((name, value))

    def add_bool_property(self, name, value):
        if False:
            while True:
                i = 10
        'Add a boolean property to the action'
        self.add_property(name, bool_text[value])

    def write(self, to_file):
        if False:
            for i in range(10):
                print('nop')
        'Write action as to a file'
        p_texts = [' '.join([self.name] + self.parameters)]
        for prop in self.properties:
            if len(prop) == 1:
                p_texts.append(prop[0])
            else:
                try:
                    p_texts.append('%s:%s' % prop)
                except:
                    raise repr(prop)
        text = ['=== ']
        text.append(' // '.join(p_texts))
        text_line = ''.join(text).encode('utf-8')
        available = 79
        while len(text_line) > available:
            to_file.write(text_line[:available])
            text_line = text_line[available:]
            to_file.write('\n... ')
            available = 79 - len('... ')
        to_file.write(text_line + '\n')

class BundleSerializerV08(BundleSerializer):

    def read(self, f):
        if False:
            while True:
                i = 10
        'Read the rest of the bundles from the supplied file.\n\n        :param f: The file to read from\n        :return: A list of bundles\n        '
        return BundleReader(f).info

    def check_compatible(self):
        if False:
            i = 10
            return i + 15
        if self.source.supports_rich_root():
            raise errors.IncompatibleBundleFormat('0.8', repr(self.source))

    def write(self, source, revision_ids, forced_bases, f):
        if False:
            while True:
                i = 10
        'Write the bundless to the supplied files.\n\n        :param source: A source for revision information\n        :param revision_ids: The list of revision ids to serialize\n        :param forced_bases: A dict of revision -> base that overrides default\n        :param f: The file to output to\n        '
        self.source = source
        self.revision_ids = revision_ids
        self.forced_bases = forced_bases
        self.to_file = f
        self.check_compatible()
        source.lock_read()
        try:
            self._write_main_header()
            pb = ui.ui_factory.nested_progress_bar()
            try:
                self._write_revisions(pb)
            finally:
                pb.finished()
        finally:
            source.unlock()

    def write_bundle(self, repository, target, base, fileobj):
        if False:
            print('Hello World!')
        return self._write_bundle(repository, target, base, fileobj)

    def _write_main_header(self):
        if False:
            while True:
                i = 10
        'Write the header for the changes'
        f = self.to_file
        f.write(_get_bundle_header('0.8'))
        f.write('#\n')

    def _write(self, key, value, indent=1, trailing_space_when_empty=False):
        if False:
            i = 10
            return i + 15
        'Write out meta information, with proper indenting, etc.\n\n        :param trailing_space_when_empty: To work around a bug in earlier\n            bundle readers, when writing an empty property, we use "prop: \n"\n            rather than writing "prop:\n".\n            If this parameter is True, and value is the empty string, we will\n            write an extra space.\n        '
        if indent < 1:
            raise ValueError('indentation must be greater than 0')
        f = self.to_file
        f.write('#' + ' ' * indent)
        f.write(key.encode('utf-8'))
        if not value:
            if trailing_space_when_empty and value == '':
                f.write(': \n')
            else:
                f.write(':\n')
        elif isinstance(value, str):
            f.write(': ')
            f.write(value)
            f.write('\n')
        elif isinstance(value, unicode):
            f.write(': ')
            f.write(value.encode('utf-8'))
            f.write('\n')
        else:
            f.write(':\n')
            for entry in value:
                f.write('#' + ' ' * (indent + 2))
                if isinstance(entry, str):
                    f.write(entry)
                else:
                    f.write(entry.encode('utf-8'))
                f.write('\n')

    def _write_revisions(self, pb):
        if False:
            return 10
        'Write the information for all of the revisions.'
        last_rev_id = None
        last_rev_tree = None
        i_max = len(self.revision_ids)
        for (i, rev_id) in enumerate(self.revision_ids):
            pb.update('Generating revision data', i, i_max)
            rev = self.source.get_revision(rev_id)
            if rev_id == last_rev_id:
                rev_tree = last_rev_tree
            else:
                rev_tree = self.source.revision_tree(rev_id)
            if rev_id in self.forced_bases:
                explicit_base = True
                base_id = self.forced_bases[rev_id]
                if base_id is None:
                    base_id = NULL_REVISION
            else:
                explicit_base = False
                if rev.parent_ids:
                    base_id = rev.parent_ids[-1]
                else:
                    base_id = NULL_REVISION
            if base_id == last_rev_id:
                base_tree = last_rev_tree
            else:
                base_tree = self.source.revision_tree(base_id)
            force_binary = i != 0
            self._write_revision(rev, rev_tree, base_id, base_tree, explicit_base, force_binary)
            last_rev_id = base_id
            last_rev_tree = base_tree

    def _testament_sha1(self, revision_id):
        if False:
            for i in range(10):
                print('nop')
        return StrictTestament.from_revision(self.source, revision_id).as_sha1()

    def _write_revision(self, rev, rev_tree, base_rev, base_tree, explicit_base, force_binary):
        if False:
            i = 10
            return i + 15
        'Write out the information for a revision.'

        def w(key, value):
            if False:
                i = 10
                return i + 15
            self._write(key, value, indent=1)
        w('message', rev.message.split('\n'))
        w('committer', rev.committer)
        w('date', format_highres_date(rev.timestamp, rev.timezone))
        self.to_file.write('\n')
        self._write_delta(rev_tree, base_tree, rev.revision_id, force_binary)
        w('revision id', rev.revision_id)
        w('sha1', self._testament_sha1(rev.revision_id))
        w('inventory sha1', rev.inventory_sha1)
        if rev.parent_ids:
            w('parent ids', rev.parent_ids)
        if explicit_base:
            w('base id', base_rev)
        if rev.properties:
            self._write('properties', None, indent=1)
            for (name, value) in sorted(rev.properties.items()):
                self._write(name, value, indent=3, trailing_space_when_empty=True)
        self.to_file.write('\n')

    def _write_action(self, name, parameters, properties=None):
        if False:
            return 10
        if properties is None:
            properties = []
        p_texts = ['%s:%s' % v for v in properties]
        self.to_file.write('=== ')
        self.to_file.write(' '.join([name] + parameters).encode('utf-8'))
        self.to_file.write(' // '.join(p_texts).encode('utf-8'))
        self.to_file.write('\n')

    def _write_delta(self, new_tree, old_tree, default_revision_id, force_binary):
        if False:
            for i in range(10):
                print('nop')
        'Write out the changes between the trees.'
        DEVNULL = '/dev/null'
        old_label = ''
        new_label = ''

        def do_diff(file_id, old_path, new_path, action, force_binary):
            if False:
                i = 10
                return i + 15

            def tree_lines(tree, require_text=False):
                if False:
                    while True:
                        i = 10
                if tree.has_id(file_id):
                    tree_file = tree.get_file(file_id)
                    if require_text is True:
                        tree_file = text_file(tree_file)
                    return tree_file.readlines()
                else:
                    return []
            try:
                if force_binary:
                    raise errors.BinaryFile()
                old_lines = tree_lines(old_tree, require_text=True)
                new_lines = tree_lines(new_tree, require_text=True)
                action.write(self.to_file)
                internal_diff(old_path, old_lines, new_path, new_lines, self.to_file)
            except errors.BinaryFile:
                old_lines = tree_lines(old_tree, require_text=False)
                new_lines = tree_lines(new_tree, require_text=False)
                action.add_property('encoding', 'base64')
                action.write(self.to_file)
                binary_diff(old_path, old_lines, new_path, new_lines, self.to_file)

        def finish_action(action, file_id, kind, meta_modified, text_modified, old_path, new_path):
            if False:
                for i in range(10):
                    print('nop')
            entry = new_tree.root_inventory[file_id]
            if entry.revision != default_revision_id:
                action.add_utf8_property('last-changed', entry.revision)
            if meta_modified:
                action.add_bool_property('executable', entry.executable)
            if text_modified and kind == 'symlink':
                action.add_property('target', entry.symlink_target)
            if text_modified and kind == 'file':
                do_diff(file_id, old_path, new_path, action, force_binary)
            else:
                action.write(self.to_file)
        delta = new_tree.changes_from(old_tree, want_unchanged=True, include_root=True)
        for (path, file_id, kind) in delta.removed:
            action = Action('removed', [kind, path]).write(self.to_file)
        for (path, file_id, kind) in delta.added:
            action = Action('added', [kind, path], [('file-id', file_id)])
            meta_modified = kind == 'file' and new_tree.is_executable(file_id)
            finish_action(action, file_id, kind, meta_modified, True, DEVNULL, path)
        for (old_path, new_path, file_id, kind, text_modified, meta_modified) in delta.renamed:
            action = Action('renamed', [kind, old_path], [(new_path,)])
            finish_action(action, file_id, kind, meta_modified, text_modified, old_path, new_path)
        for (path, file_id, kind, text_modified, meta_modified) in delta.modified:
            action = Action('modified', [kind, path])
            finish_action(action, file_id, kind, meta_modified, text_modified, path, path)
        for (path, file_id, kind) in delta.unchanged:
            new_rev = new_tree.get_file_revision(file_id)
            if new_rev is None:
                continue
            old_rev = old_tree.get_file_revision(file_id)
            if new_rev != old_rev:
                action = Action('modified', [new_tree.kind(file_id), new_tree.id2path(file_id)])
                action.add_utf8_property('last-changed', new_rev)
                action.write(self.to_file)

class BundleReader(object):
    """This class reads in a bundle from a file, and returns
    a Bundle object, which can then be applied against a tree.
    """

    def __init__(self, from_file):
        if False:
            i = 10
            return i + 15
        'Read in the bundle from the file.\n\n        :param from_file: A file-like object (must have iterator support).\n        '
        object.__init__(self)
        self.from_file = iter(from_file)
        self._next_line = None
        self.info = self._get_info()
        self._read()
        self._validate()

    def _get_info(self):
        if False:
            while True:
                i = 10
        return BundleInfo08()

    def _read(self):
        if False:
            print('Hello World!')
        self._next().next()
        while self._next_line is not None:
            if not self._read_revision_header():
                break
            if self._next_line is None:
                break
            self._read_patches()
            self._read_footer()

    def _validate(self):
        if False:
            return 10
        'Make sure that the information read in makes sense\n        and passes appropriate checksums.\n        '
        self.info.complete_info()

    def _next(self):
        if False:
            i = 10
            return i + 15
        'yield the next line, but secretly\n        keep 1 extra line for peeking.\n        '
        for line in self.from_file:
            last = self._next_line
            self._next_line = line
            if last is not None:
                yield last
        last = self._next_line
        self._next_line = None
        yield last

    def _read_revision_header(self):
        if False:
            i = 10
            return i + 15
        found_something = False
        self.info.revisions.append(RevisionInfo(None))
        for line in self._next():
            if line is None or line == '\n':
                break
            if not line.startswith('#'):
                continue
            found_something = True
            self._handle_next(line)
        if not found_something:
            self.info.revisions.pop()
        return found_something

    def _read_next_entry(self, line, indent=1):
        if False:
            while True:
                i = 10
        'Read in a key-value pair\n        '
        if not line.startswith('#'):
            raise errors.MalformedHeader('Bzr header did not start with #')
        line = line[1:-1].decode('utf-8')
        if line[:indent] == ' ' * indent:
            line = line[indent:]
        if not line:
            return (None, None)
        loc = line.find(': ')
        if loc != -1:
            key = line[:loc]
            value = line[loc + 2:]
            if not value:
                value = self._read_many(indent=indent + 2)
        elif line[-1:] == ':':
            key = line[:-1]
            value = self._read_many(indent=indent + 2)
        else:
            raise errors.MalformedHeader('While looking for key: value pairs, did not find the colon %r' % line)
        key = key.replace(' ', '_')
        return (key, value)

    def _handle_next(self, line):
        if False:
            return 10
        if line is None:
            return
        (key, value) = self._read_next_entry(line, indent=1)
        mutter('_handle_next %r => %r' % (key, value))
        if key is None:
            return
        revision_info = self.info.revisions[-1]
        if key in revision_info.__dict__:
            if getattr(revision_info, key) is None:
                if key in ('file_id', 'revision_id', 'base_id'):
                    value = value.encode('utf8')
                elif key in 'parent_ids':
                    value = [v.encode('utf8') for v in value]
                setattr(revision_info, key, value)
            else:
                raise errors.MalformedHeader('Duplicated Key: %s' % key)
        else:
            raise errors.MalformedHeader('Unknown Key: "%s"' % key)

    def _read_many(self, indent):
        if False:
            i = 10
            return i + 15
        'If a line ends with no entry, that means that it should be\n        followed with multiple lines of values.\n\n        This detects the end of the list, because it will be a line that\n        does not start properly indented.\n        '
        values = []
        start = '#' + ' ' * indent
        if self._next_line is None or self._next_line[:len(start)] != start:
            return values
        for line in self._next():
            values.append(line[len(start):-1].decode('utf-8'))
            if self._next_line is None or self._next_line[:len(start)] != start:
                break
        return values

    def _read_one_patch(self):
        if False:
            print('Hello World!')
        'Read in one patch, return the complete patch, along with\n        the next line.\n\n        :return: action, lines, do_continue\n        '
        if self._next_line is None or self._next_line.startswith('#'):
            return (None, [], False)
        first = True
        lines = []
        for line in self._next():
            if first:
                if not line.startswith('==='):
                    raise errors.MalformedPatches('The first line of all patches should be a bzr meta line "===": %r' % line)
                action = line[4:-1].decode('utf-8')
            elif line.startswith('... '):
                action += line[len('... '):-1].decode('utf-8')
            if self._next_line is not None and self._next_line.startswith('==='):
                return (action, lines, True)
            elif self._next_line is None or self._next_line.startswith('#'):
                return (action, lines, False)
            if first:
                first = False
            elif not line.startswith('... '):
                lines.append(line)
        return (action, lines, False)

    def _read_patches(self):
        if False:
            print('Hello World!')
        do_continue = True
        revision_actions = []
        while do_continue:
            (action, lines, do_continue) = self._read_one_patch()
            if action is not None:
                revision_actions.append((action, lines))
        if self.info.revisions[-1].tree_actions is not None:
            raise AssertionError()
        self.info.revisions[-1].tree_actions = revision_actions

    def _read_footer(self):
        if False:
            i = 10
            return i + 15
        'Read the rest of the meta information.\n\n        :param first_line:  The previous step iterates past what it\n                            can handle. That extra line is given here.\n        '
        for line in self._next():
            self._handle_next(line)
            if self._next_line is None:
                break
            if not self._next_line.startswith('#'):
                self._next().next()
                break

class BundleInfo08(BundleInfo):

    def _update_tree(self, bundle_tree, revision_id):
        if False:
            while True:
                i = 10
        bundle_tree.note_last_changed('', revision_id)
        BundleInfo._update_tree(self, bundle_tree, revision_id)

    def _testament_sha1_from_revision(self, repository, revision_id):
        if False:
            return 10
        testament = StrictTestament.from_revision(repository, revision_id)
        return testament.as_sha1()

    def _testament_sha1(self, revision, tree):
        if False:
            for i in range(10):
                print('nop')
        return StrictTestament(revision, tree).as_sha1()