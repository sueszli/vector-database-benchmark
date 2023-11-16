"""Knit-based pack repository formats."""
from __future__ import absolute_import
from bzrlib.lazy_import import lazy_import
lazy_import(globals(), '\nfrom itertools import izip\nimport time\n\nfrom bzrlib import (\n    controldir,\n    debug,\n    errors,\n    knit,\n    osutils,\n    pack,\n    revision as _mod_revision,\n    trace,\n    tsort,\n    ui,\n    xml5,\n    xml6,\n    xml7,\n    )\nfrom bzrlib.knit import (\n    _KnitGraphIndex,\n    KnitPlainFactory,\n    KnitVersionedFiles,\n    )\n')
from bzrlib import btree_index
from bzrlib.index import CombinedGraphIndex, GraphIndex, GraphIndexPrefixAdapter, InMemoryGraphIndex
from bzrlib.repofmt.knitrepo import KnitRepository
from bzrlib.repofmt.pack_repo import _DirectPackAccess, NewPack, RepositoryFormatPack, ResumedPack, Packer, PackCommitBuilder, PackRepository, PackRootCommitBuilder, RepositoryPackCollection
from bzrlib.vf_repository import StreamSource

class KnitPackRepository(PackRepository, KnitRepository):

    def __init__(self, _format, a_bzrdir, control_files, _commit_builder_class, _serializer):
        if False:
            print('Hello World!')
        PackRepository.__init__(self, _format, a_bzrdir, control_files, _commit_builder_class, _serializer)
        if self._format.supports_chks:
            raise AssertionError('chk not supported')
        index_transport = self._transport.clone('indices')
        self._pack_collection = KnitRepositoryPackCollection(self, self._transport, index_transport, self._transport.clone('upload'), self._transport.clone('packs'), _format.index_builder_class, _format.index_class, use_chk_index=False)
        self.inventories = KnitVersionedFiles(_KnitGraphIndex(self._pack_collection.inventory_index.combined_index, add_callback=self._pack_collection.inventory_index.add_callback, deltas=True, parents=True, is_locked=self.is_locked), data_access=self._pack_collection.inventory_index.data_access, max_delta_chain=200)
        self.revisions = KnitVersionedFiles(_KnitGraphIndex(self._pack_collection.revision_index.combined_index, add_callback=self._pack_collection.revision_index.add_callback, deltas=False, parents=True, is_locked=self.is_locked, track_external_parent_refs=True), data_access=self._pack_collection.revision_index.data_access, max_delta_chain=0)
        self.signatures = KnitVersionedFiles(_KnitGraphIndex(self._pack_collection.signature_index.combined_index, add_callback=self._pack_collection.signature_index.add_callback, deltas=False, parents=False, is_locked=self.is_locked), data_access=self._pack_collection.signature_index.data_access, max_delta_chain=0)
        self.texts = KnitVersionedFiles(_KnitGraphIndex(self._pack_collection.text_index.combined_index, add_callback=self._pack_collection.text_index.add_callback, deltas=True, parents=True, is_locked=self.is_locked), data_access=self._pack_collection.text_index.data_access, max_delta_chain=200)
        self.chk_bytes = None
        self._write_lock_count = 0
        self._transaction = None
        self._reconcile_does_inventory_gc = True
        self._reconcile_fixes_text_parents = True
        self._reconcile_backsup_inventory = False

    def _get_source(self, to_format):
        if False:
            while True:
                i = 10
        if to_format.network_name() == self._format.network_name():
            return KnitPackStreamSource(self, to_format)
        return PackRepository._get_source(self, to_format)

    def _reconcile_pack(self, collection, packs, extension, revs, pb):
        if False:
            while True:
                i = 10
        packer = KnitReconcilePacker(collection, packs, extension, revs)
        return packer.pack(pb)

class RepositoryFormatKnitPack1(RepositoryFormatPack):
    """A no-subtrees parameterized Pack repository.

    This format was introduced in 0.92.
    """
    repository_class = KnitPackRepository
    _commit_builder_class = PackCommitBuilder

    @property
    def _serializer(self):
        if False:
            print('Hello World!')
        return xml5.serializer_v5
    index_builder_class = InMemoryGraphIndex
    index_class = GraphIndex

    def _get_matching_bzrdir(self):
        if False:
            print('Hello World!')
        return controldir.format_registry.make_bzrdir('pack-0.92')

    def _ignore_setting_bzrdir(self, format):
        if False:
            for i in range(10):
                print('nop')
        pass
    _matchingbzrdir = property(_get_matching_bzrdir, _ignore_setting_bzrdir)

    @classmethod
    def get_format_string(cls):
        if False:
            print('Hello World!')
        'See RepositoryFormat.get_format_string().'
        return 'Bazaar pack repository format 1 (needs bzr 0.92)\n'

    def get_format_description(self):
        if False:
            print('Hello World!')
        'See RepositoryFormat.get_format_description().'
        return 'Packs containing knits without subtree support'

class RepositoryFormatKnitPack3(RepositoryFormatPack):
    """A subtrees parameterized Pack repository.

    This repository format uses the xml7 serializer to get:
     - support for recording full info about the tree root
     - support for recording tree-references

    This format was introduced in 0.92.
    """
    repository_class = KnitPackRepository
    _commit_builder_class = PackRootCommitBuilder
    rich_root_data = True
    experimental = True
    supports_tree_reference = True

    @property
    def _serializer(self):
        if False:
            return 10
        return xml7.serializer_v7
    index_builder_class = InMemoryGraphIndex
    index_class = GraphIndex

    def _get_matching_bzrdir(self):
        if False:
            return 10
        return controldir.format_registry.make_bzrdir('pack-0.92-subtree')

    def _ignore_setting_bzrdir(self, format):
        if False:
            i = 10
            return i + 15
        pass
    _matchingbzrdir = property(_get_matching_bzrdir, _ignore_setting_bzrdir)

    @classmethod
    def get_format_string(cls):
        if False:
            while True:
                i = 10
        'See RepositoryFormat.get_format_string().'
        return 'Bazaar pack repository format 1 with subtree support (needs bzr 0.92)\n'

    def get_format_description(self):
        if False:
            for i in range(10):
                print('nop')
        'See RepositoryFormat.get_format_description().'
        return 'Packs containing knits with subtree support\n'

class RepositoryFormatKnitPack4(RepositoryFormatPack):
    """A rich-root, no subtrees parameterized Pack repository.

    This repository format uses the xml6 serializer to get:
     - support for recording full info about the tree root

    This format was introduced in 1.0.
    """
    repository_class = KnitPackRepository
    _commit_builder_class = PackRootCommitBuilder
    rich_root_data = True
    supports_tree_reference = False

    @property
    def _serializer(self):
        if False:
            i = 10
            return i + 15
        return xml6.serializer_v6
    index_builder_class = InMemoryGraphIndex
    index_class = GraphIndex

    def _get_matching_bzrdir(self):
        if False:
            print('Hello World!')
        return controldir.format_registry.make_bzrdir('rich-root-pack')

    def _ignore_setting_bzrdir(self, format):
        if False:
            print('Hello World!')
        pass
    _matchingbzrdir = property(_get_matching_bzrdir, _ignore_setting_bzrdir)

    @classmethod
    def get_format_string(cls):
        if False:
            print('Hello World!')
        'See RepositoryFormat.get_format_string().'
        return 'Bazaar pack repository format 1 with rich root (needs bzr 1.0)\n'

    def get_format_description(self):
        if False:
            for i in range(10):
                print('nop')
        'See RepositoryFormat.get_format_description().'
        return 'Packs containing knits with rich root support\n'

class RepositoryFormatKnitPack5(RepositoryFormatPack):
    """Repository that supports external references to allow stacking.

    New in release 1.6.

    Supports external lookups, which results in non-truncated ghosts after
    reconcile compared to pack-0.92 formats.
    """
    repository_class = KnitPackRepository
    _commit_builder_class = PackCommitBuilder
    supports_external_lookups = True
    index_builder_class = InMemoryGraphIndex
    index_class = GraphIndex

    @property
    def _serializer(self):
        if False:
            return 10
        return xml5.serializer_v5

    def _get_matching_bzrdir(self):
        if False:
            while True:
                i = 10
        return controldir.format_registry.make_bzrdir('1.6')

    def _ignore_setting_bzrdir(self, format):
        if False:
            return 10
        pass
    _matchingbzrdir = property(_get_matching_bzrdir, _ignore_setting_bzrdir)

    @classmethod
    def get_format_string(cls):
        if False:
            i = 10
            return i + 15
        'See RepositoryFormat.get_format_string().'
        return 'Bazaar RepositoryFormatKnitPack5 (bzr 1.6)\n'

    def get_format_description(self):
        if False:
            print('Hello World!')
        'See RepositoryFormat.get_format_description().'
        return 'Packs 5 (adds stacking support, requires bzr 1.6)'

class RepositoryFormatKnitPack5RichRoot(RepositoryFormatPack):
    """A repository with rich roots and stacking.

    New in release 1.6.1.

    Supports stacking on other repositories, allowing data to be accessed
    without being stored locally.
    """
    repository_class = KnitPackRepository
    _commit_builder_class = PackRootCommitBuilder
    rich_root_data = True
    supports_tree_reference = False
    supports_external_lookups = True
    index_builder_class = InMemoryGraphIndex
    index_class = GraphIndex

    @property
    def _serializer(self):
        if False:
            for i in range(10):
                print('nop')
        return xml6.serializer_v6

    def _get_matching_bzrdir(self):
        if False:
            i = 10
            return i + 15
        return controldir.format_registry.make_bzrdir('1.6.1-rich-root')

    def _ignore_setting_bzrdir(self, format):
        if False:
            return 10
        pass
    _matchingbzrdir = property(_get_matching_bzrdir, _ignore_setting_bzrdir)

    @classmethod
    def get_format_string(cls):
        if False:
            while True:
                i = 10
        'See RepositoryFormat.get_format_string().'
        return 'Bazaar RepositoryFormatKnitPack5RichRoot (bzr 1.6.1)\n'

    def get_format_description(self):
        if False:
            i = 10
            return i + 15
        return 'Packs 5 rich-root (adds stacking support, requires bzr 1.6.1)'

class RepositoryFormatKnitPack5RichRootBroken(RepositoryFormatPack):
    """A repository with rich roots and external references.

    New in release 1.6.

    Supports external lookups, which results in non-truncated ghosts after
    reconcile compared to pack-0.92 formats.

    This format was deprecated because the serializer it uses accidentally
    supported subtrees, when the format was not intended to. This meant that
    someone could accidentally fetch from an incorrect repository.
    """
    repository_class = KnitPackRepository
    _commit_builder_class = PackRootCommitBuilder
    rich_root_data = True
    supports_tree_reference = False
    supports_external_lookups = True
    index_builder_class = InMemoryGraphIndex
    index_class = GraphIndex

    @property
    def _serializer(self):
        if False:
            return 10
        return xml7.serializer_v7

    def _get_matching_bzrdir(self):
        if False:
            while True:
                i = 10
        matching = controldir.format_registry.make_bzrdir('1.6.1-rich-root')
        matching.repository_format = self
        return matching

    def _ignore_setting_bzrdir(self, format):
        if False:
            for i in range(10):
                print('nop')
        pass
    _matchingbzrdir = property(_get_matching_bzrdir, _ignore_setting_bzrdir)

    @classmethod
    def get_format_string(cls):
        if False:
            for i in range(10):
                print('nop')
        'See RepositoryFormat.get_format_string().'
        return 'Bazaar RepositoryFormatKnitPack5RichRoot (bzr 1.6)\n'

    def get_format_description(self):
        if False:
            return 10
        return 'Packs 5 rich-root (adds stacking support, requires bzr 1.6) (deprecated)'

    def is_deprecated(self):
        if False:
            while True:
                i = 10
        return True

class RepositoryFormatKnitPack6(RepositoryFormatPack):
    """A repository with stacking and btree indexes,
    without rich roots or subtrees.

    This is equivalent to pack-1.6 with B+Tree indices.
    """
    repository_class = KnitPackRepository
    _commit_builder_class = PackCommitBuilder
    supports_external_lookups = True
    index_builder_class = btree_index.BTreeBuilder
    index_class = btree_index.BTreeGraphIndex

    @property
    def _serializer(self):
        if False:
            print('Hello World!')
        return xml5.serializer_v5

    def _get_matching_bzrdir(self):
        if False:
            return 10
        return controldir.format_registry.make_bzrdir('1.9')

    def _ignore_setting_bzrdir(self, format):
        if False:
            i = 10
            return i + 15
        pass
    _matchingbzrdir = property(_get_matching_bzrdir, _ignore_setting_bzrdir)

    @classmethod
    def get_format_string(cls):
        if False:
            return 10
        'See RepositoryFormat.get_format_string().'
        return 'Bazaar RepositoryFormatKnitPack6 (bzr 1.9)\n'

    def get_format_description(self):
        if False:
            while True:
                i = 10
        'See RepositoryFormat.get_format_description().'
        return 'Packs 6 (uses btree indexes, requires bzr 1.9)'

class RepositoryFormatKnitPack6RichRoot(RepositoryFormatPack):
    """A repository with rich roots, no subtrees, stacking and btree indexes.

    1.6-rich-root with B+Tree indices.
    """
    repository_class = KnitPackRepository
    _commit_builder_class = PackRootCommitBuilder
    rich_root_data = True
    supports_tree_reference = False
    supports_external_lookups = True
    index_builder_class = btree_index.BTreeBuilder
    index_class = btree_index.BTreeGraphIndex

    @property
    def _serializer(self):
        if False:
            while True:
                i = 10
        return xml6.serializer_v6

    def _get_matching_bzrdir(self):
        if False:
            return 10
        return controldir.format_registry.make_bzrdir('1.9-rich-root')

    def _ignore_setting_bzrdir(self, format):
        if False:
            return 10
        pass
    _matchingbzrdir = property(_get_matching_bzrdir, _ignore_setting_bzrdir)

    @classmethod
    def get_format_string(cls):
        if False:
            print('Hello World!')
        'See RepositoryFormat.get_format_string().'
        return 'Bazaar RepositoryFormatKnitPack6RichRoot (bzr 1.9)\n'

    def get_format_description(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Packs 6 rich-root (uses btree indexes, requires bzr 1.9)'

class RepositoryFormatPackDevelopment2Subtree(RepositoryFormatPack):
    """A subtrees development repository.

    This format should be retained in 2.3, to provide an upgrade path from this
    to RepositoryFormat2aSubtree.  It can be removed in later releases.

    1.6.1-subtree[as it might have been] with B+Tree indices.
    """
    repository_class = KnitPackRepository
    _commit_builder_class = PackRootCommitBuilder
    rich_root_data = True
    experimental = True
    supports_tree_reference = True
    supports_external_lookups = True
    index_builder_class = btree_index.BTreeBuilder
    index_class = btree_index.BTreeGraphIndex

    @property
    def _serializer(self):
        if False:
            while True:
                i = 10
        return xml7.serializer_v7

    def _get_matching_bzrdir(self):
        if False:
            for i in range(10):
                print('nop')
        return controldir.format_registry.make_bzrdir('development5-subtree')

    def _ignore_setting_bzrdir(self, format):
        if False:
            for i in range(10):
                print('nop')
        pass
    _matchingbzrdir = property(_get_matching_bzrdir, _ignore_setting_bzrdir)

    @classmethod
    def get_format_string(cls):
        if False:
            for i in range(10):
                print('nop')
        'See RepositoryFormat.get_format_string().'
        return 'Bazaar development format 2 with subtree support (needs bzr.dev from before 1.8)\n'

    def get_format_description(self):
        if False:
            print('Hello World!')
        'See RepositoryFormat.get_format_description().'
        return 'Development repository format, currently the same as 1.6.1-subtree with B+Tree indices.\n'

class KnitPackStreamSource(StreamSource):
    """A StreamSource used to transfer data between same-format KnitPack repos.

    This source assumes:
        1) Same serialization format for all objects
        2) Same root information
        3) XML format inventories
        4) Atomic inserts (so we can stream inventory texts before text
           content)
        5) No chk_bytes
    """

    def __init__(self, from_repository, to_format):
        if False:
            for i in range(10):
                print('nop')
        super(KnitPackStreamSource, self).__init__(from_repository, to_format)
        self._text_keys = None
        self._text_fetch_order = 'unordered'

    def _get_filtered_inv_stream(self, revision_ids):
        if False:
            for i in range(10):
                print('nop')
        from_repo = self.from_repository
        parent_ids = from_repo._find_parent_ids_of_revisions(revision_ids)
        parent_keys = [(p,) for p in parent_ids]
        find_text_keys = from_repo._serializer._find_text_key_references
        parent_text_keys = set(find_text_keys(from_repo._inventory_xml_lines_for_keys(parent_keys)))
        content_text_keys = set()
        knit = KnitVersionedFiles(None, None)
        factory = KnitPlainFactory()

        def find_text_keys_from_content(record):
            if False:
                return 10
            if record.storage_kind not in ('knit-delta-gz', 'knit-ft-gz'):
                raise ValueError('Unknown content storage kind for inventory text: %s' % (record.storage_kind,))
            raw_data = record._raw_record
            revision_id = record.key[-1]
            (content, _) = knit._parse_record(revision_id, raw_data)
            if record.storage_kind == 'knit-delta-gz':
                line_iterator = factory.get_linedelta_content(content)
            elif record.storage_kind == 'knit-ft-gz':
                line_iterator = factory.get_fulltext_content(content)
            content_text_keys.update(find_text_keys([(line, revision_id) for line in line_iterator]))
        revision_keys = [(r,) for r in revision_ids]

        def _filtered_inv_stream():
            if False:
                while True:
                    i = 10
            source_vf = from_repo.inventories
            stream = source_vf.get_record_stream(revision_keys, 'unordered', False)
            for record in stream:
                if record.storage_kind == 'absent':
                    raise errors.NoSuchRevision(from_repo, record.key)
                find_text_keys_from_content(record)
                yield record
            self._text_keys = content_text_keys - parent_text_keys
        return ('inventories', _filtered_inv_stream())

    def _get_text_stream(self):
        if False:
            i = 10
            return i + 15
        text_stream = self.from_repository.texts.get_record_stream(self._text_keys, self._text_fetch_order, False)
        return ('texts', text_stream)

    def get_stream(self, search):
        if False:
            while True:
                i = 10
        revision_ids = search.get_keys()
        for stream_info in self._fetch_revision_texts(revision_ids):
            yield stream_info
        self._revision_keys = [(rev_id,) for rev_id in revision_ids]
        yield self._get_filtered_inv_stream(revision_ids)
        yield self._get_text_stream()

class KnitPacker(Packer):
    """Packer that works with knit packs."""

    def __init__(self, pack_collection, packs, suffix, revision_ids=None, reload_func=None):
        if False:
            while True:
                i = 10
        super(KnitPacker, self).__init__(pack_collection, packs, suffix, revision_ids=revision_ids, reload_func=reload_func)

    def _pack_map_and_index_list(self, index_attribute):
        if False:
            print('Hello World!')
        'Convert a list of packs to an index pack map and index list.\n\n        :param index_attribute: The attribute that the desired index is found\n            on.\n        :return: A tuple (map, list) where map contains the dict from\n            index:pack_tuple, and list contains the indices in the preferred\n            access order.\n        '
        indices = []
        pack_map = {}
        for pack_obj in self.packs:
            index = getattr(pack_obj, index_attribute)
            indices.append(index)
            pack_map[index] = pack_obj
        return (pack_map, indices)

    def _index_contents(self, indices, key_filter=None):
        if False:
            print('Hello World!')
        'Get an iterable of the index contents from a pack_map.\n\n        :param indices: The list of indices to query\n        :param key_filter: An optional filter to limit the keys returned.\n        '
        all_index = CombinedGraphIndex(indices)
        if key_filter is None:
            return all_index.iter_all_entries()
        else:
            return all_index.iter_entries(key_filter)

    def _copy_nodes(self, nodes, index_map, writer, write_index, output_lines=None):
        if False:
            for i in range(10):
                print('nop')
        'Copy knit nodes between packs with no graph references.\n\n        :param output_lines: Output full texts of copied items.\n        '
        pb = ui.ui_factory.nested_progress_bar()
        try:
            return self._do_copy_nodes(nodes, index_map, writer, write_index, pb, output_lines=output_lines)
        finally:
            pb.finished()

    def _do_copy_nodes(self, nodes, index_map, writer, write_index, pb, output_lines=None):
        if False:
            for i in range(10):
                print('nop')
        knit = KnitVersionedFiles(None, None)
        nodes = sorted(nodes)
        request_groups = {}
        for (index, key, value) in nodes:
            if index not in request_groups:
                request_groups[index] = []
            request_groups[index].append((key, value))
        record_index = 0
        pb.update('Copied record', record_index, len(nodes))
        for (index, items) in request_groups.iteritems():
            pack_readv_requests = []
            for (key, value) in items:
                bits = value[1:].split(' ')
                (offset, length) = (int(bits[0]), int(bits[1]))
                pack_readv_requests.append((offset, length, (key, value[0])))
            pack_readv_requests.sort()
            pack_obj = index_map[index]
            (transport, path) = pack_obj.access_tuple()
            try:
                reader = pack.make_readv_reader(transport, path, [offset[0:2] for offset in pack_readv_requests])
            except errors.NoSuchFile:
                if self._reload_func is not None:
                    self._reload_func()
                raise
            for ((names, read_func), (_1, _2, (key, eol_flag))) in izip(reader.iter_records(), pack_readv_requests):
                raw_data = read_func(None)
                if output_lines is not None:
                    output_lines(knit._parse_record(key[-1], raw_data)[0])
                else:
                    (df, _) = knit._parse_record_header(key, raw_data)
                    df.close()
                (pos, size) = writer.add_bytes_record(raw_data, names)
                write_index.add_node(key, eol_flag + '%d %d' % (pos, size))
                pb.update('Copied record', record_index)
                record_index += 1

    def _copy_nodes_graph(self, index_map, writer, write_index, readv_group_iter, total_items, output_lines=False):
        if False:
            i = 10
            return i + 15
        'Copy knit nodes between packs.\n\n        :param output_lines: Return lines present in the copied data as\n            an iterator of line,version_id.\n        '
        pb = ui.ui_factory.nested_progress_bar()
        try:
            for result in self._do_copy_nodes_graph(index_map, writer, write_index, output_lines, pb, readv_group_iter, total_items):
                yield result
        except Exception:
            pb.finished()
            raise
        else:
            pb.finished()

    def _do_copy_nodes_graph(self, index_map, writer, write_index, output_lines, pb, readv_group_iter, total_items):
        if False:
            return 10
        knit = KnitVersionedFiles(None, None)
        if output_lines:
            factory = KnitPlainFactory()
        record_index = 0
        pb.update('Copied record', record_index, total_items)
        for (index, readv_vector, node_vector) in readv_group_iter:
            pack_obj = index_map[index]
            (transport, path) = pack_obj.access_tuple()
            try:
                reader = pack.make_readv_reader(transport, path, readv_vector)
            except errors.NoSuchFile:
                if self._reload_func is not None:
                    self._reload_func()
                raise
            for ((names, read_func), (key, eol_flag, references)) in izip(reader.iter_records(), node_vector):
                raw_data = read_func(None)
                if output_lines:
                    (content, _) = knit._parse_record(key[-1], raw_data)
                    if len(references[-1]) == 0:
                        line_iterator = factory.get_fulltext_content(content)
                    else:
                        line_iterator = factory.get_linedelta_content(content)
                    for line in line_iterator:
                        yield (line, key)
                else:
                    (df, _) = knit._parse_record_header(key, raw_data)
                    df.close()
                (pos, size) = writer.add_bytes_record(raw_data, names)
                write_index.add_node(key, eol_flag + '%d %d' % (pos, size), references)
                pb.update('Copied record', record_index)
                record_index += 1

    def _process_inventory_lines(self, inv_lines):
        if False:
            for i in range(10):
                print('nop')
        'Use up the inv_lines generator and setup a text key filter.'
        repo = self._pack_collection.repo
        fileid_revisions = repo._find_file_ids_from_xml_inventory_lines(inv_lines, self.revision_keys)
        text_filter = []
        for (fileid, file_revids) in fileid_revisions.iteritems():
            text_filter.extend([(fileid, file_revid) for file_revid in file_revids])
        self._text_filter = text_filter

    def _copy_inventory_texts(self):
        if False:
            return 10
        inv_keys = self._revision_keys
        (inventory_index_map, inventory_indices) = self._pack_map_and_index_list('inventory_index')
        inv_nodes = self._index_contents(inventory_indices, inv_keys)
        self.pb.update('Copying inventory texts', 2)
        (total_items, readv_group_iter) = self._least_readv_node_readv(inv_nodes)
        output_lines = bool(self.revision_ids)
        inv_lines = self._copy_nodes_graph(inventory_index_map, self.new_pack._writer, self.new_pack.inventory_index, readv_group_iter, total_items, output_lines=output_lines)
        if self.revision_ids:
            self._process_inventory_lines(inv_lines)
        else:
            list(inv_lines)
            self._text_filter = None
        if 'pack' in debug.debug_flags:
            trace.mutter('%s: create_pack: inventories copied: %s%s %d items t+%6.3fs', time.ctime(), self._pack_collection._upload_transport.base, self.new_pack.random_name, self.new_pack.inventory_index.key_count(), time.time() - self.new_pack.start_time)

    def _update_pack_order(self, entries, index_to_pack_map):
        if False:
            i = 10
            return i + 15
        "Determine how we want our packs to be ordered.\n\n        This changes the sort order of the self.packs list so that packs unused\n        by 'entries' will be at the end of the list, so that future requests\n        can avoid probing them.  Used packs will be at the front of the\n        self.packs list, in the order of their first use in 'entries'.\n\n        :param entries: A list of (index, ...) tuples\n        :param index_to_pack_map: A mapping from index objects to pack objects.\n        "
        packs = []
        seen_indexes = set()
        for entry in entries:
            index = entry[0]
            if index not in seen_indexes:
                packs.append(index_to_pack_map[index])
                seen_indexes.add(index)
        if len(packs) == len(self.packs):
            if 'pack' in debug.debug_flags:
                trace.mutter('Not changing pack list, all packs used.')
            return
        seen_packs = set(packs)
        for pack in self.packs:
            if pack not in seen_packs:
                packs.append(pack)
                seen_packs.add(pack)
        if 'pack' in debug.debug_flags:
            old_names = [p.access_tuple()[1] for p in self.packs]
            new_names = [p.access_tuple()[1] for p in packs]
            trace.mutter('Reordering packs\nfrom: %s\n  to: %s', old_names, new_names)
        self.packs = packs

    def _copy_revision_texts(self):
        if False:
            for i in range(10):
                print('nop')
        if self.revision_ids:
            revision_keys = [(revision_id,) for revision_id in self.revision_ids]
        else:
            revision_keys = None
        (revision_index_map, revision_indices) = self._pack_map_and_index_list('revision_index')
        revision_nodes = self._index_contents(revision_indices, revision_keys)
        revision_nodes = list(revision_nodes)
        self._update_pack_order(revision_nodes, revision_index_map)
        self.pb.update('Copying revision texts', 1)
        (total_items, readv_group_iter) = self._revision_node_readv(revision_nodes)
        list(self._copy_nodes_graph(revision_index_map, self.new_pack._writer, self.new_pack.revision_index, readv_group_iter, total_items))
        if 'pack' in debug.debug_flags:
            trace.mutter('%s: create_pack: revisions copied: %s%s %d items t+%6.3fs', time.ctime(), self._pack_collection._upload_transport.base, self.new_pack.random_name, self.new_pack.revision_index.key_count(), time.time() - self.new_pack.start_time)
        self._revision_keys = revision_keys

    def _get_text_nodes(self):
        if False:
            while True:
                i = 10
        (text_index_map, text_indices) = self._pack_map_and_index_list('text_index')
        return (text_index_map, self._index_contents(text_indices, self._text_filter))

    def _copy_text_texts(self):
        if False:
            while True:
                i = 10
        (text_index_map, text_nodes) = self._get_text_nodes()
        if self._text_filter is not None:
            text_nodes = set(text_nodes)
            present_text_keys = set((_node[1] for _node in text_nodes))
            missing_text_keys = set(self._text_filter) - present_text_keys
            if missing_text_keys:
                trace.mutter('missing keys during fetch: %r', missing_text_keys)
                a_missing_key = missing_text_keys.pop()
                raise errors.RevisionNotPresent(a_missing_key[1], a_missing_key[0])
        self.pb.update('Copying content texts', 3)
        (total_items, readv_group_iter) = self._least_readv_node_readv(text_nodes)
        list(self._copy_nodes_graph(text_index_map, self.new_pack._writer, self.new_pack.text_index, readv_group_iter, total_items))
        self._log_copied_texts()

    def _create_pack_from_packs(self):
        if False:
            while True:
                i = 10
        self.pb.update('Opening pack', 0, 5)
        self.new_pack = self.open_pack()
        new_pack = self.new_pack
        new_pack.set_write_cache_size(1024 * 1024)
        if 'pack' in debug.debug_flags:
            plain_pack_list = ['%s%s' % (a_pack.pack_transport.base, a_pack.name) for a_pack in self.packs]
            if self.revision_ids is not None:
                rev_count = len(self.revision_ids)
            else:
                rev_count = 'all'
            trace.mutter('%s: create_pack: creating pack from source packs: %s%s %s revisions wanted %s t=0', time.ctime(), self._pack_collection._upload_transport.base, new_pack.random_name, plain_pack_list, rev_count)
        self._copy_revision_texts()
        self._copy_inventory_texts()
        self._copy_text_texts()
        signature_filter = self._revision_keys
        (signature_index_map, signature_indices) = self._pack_map_and_index_list('signature_index')
        signature_nodes = self._index_contents(signature_indices, signature_filter)
        self.pb.update('Copying signature texts', 4)
        self._copy_nodes(signature_nodes, signature_index_map, new_pack._writer, new_pack.signature_index)
        if 'pack' in debug.debug_flags:
            trace.mutter('%s: create_pack: revision signatures copied: %s%s %d items t+%6.3fs', time.ctime(), self._pack_collection._upload_transport.base, new_pack.random_name, new_pack.signature_index.key_count(), time.time() - new_pack.start_time)
        new_pack._check_references()
        if not self._use_pack(new_pack):
            new_pack.abort()
            return None
        self.pb.update('Finishing pack', 5)
        new_pack.finish()
        self._pack_collection.allocate(new_pack)
        return new_pack

    def _least_readv_node_readv(self, nodes):
        if False:
            while True:
                i = 10
        "Generate request groups for nodes using the least readv's.\n\n        :param nodes: An iterable of graph index nodes.\n        :return: Total node count and an iterator of the data needed to perform\n            readvs to obtain the data for nodes. Each item yielded by the\n            iterator is a tuple with:\n            index, readv_vector, node_vector. readv_vector is a list ready to\n            hand to the transport readv method, and node_vector is a list of\n            (key, eol_flag, references) for the node retrieved by the\n            matching readv_vector.\n        "
        nodes = sorted(nodes)
        total = len(nodes)
        request_groups = {}
        for (index, key, value, references) in nodes:
            if index not in request_groups:
                request_groups[index] = []
            request_groups[index].append((key, value, references))
        result = []
        for (index, items) in request_groups.iteritems():
            pack_readv_requests = []
            for (key, value, references) in items:
                bits = value[1:].split(' ')
                (offset, length) = (int(bits[0]), int(bits[1]))
                pack_readv_requests.append(((offset, length), (key, value[0], references)))
            pack_readv_requests.sort()
            pack_readv = [readv for (readv, node) in pack_readv_requests]
            node_vector = [node for (readv, node) in pack_readv_requests]
            result.append((index, pack_readv, node_vector))
        return (total, result)

    def _revision_node_readv(self, revision_nodes):
        if False:
            print('Hello World!')
        "Return the total revisions and the readv's to issue.\n\n        :param revision_nodes: The revision index contents for the packs being\n            incorporated into the new pack.\n        :return: As per _least_readv_node_readv.\n        "
        return self._least_readv_node_readv(revision_nodes)

class KnitReconcilePacker(KnitPacker):
    """A packer which regenerates indices etc as it copies.

    This is used by ``bzr reconcile`` to cause parent text pointers to be
    regenerated.
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(KnitReconcilePacker, self).__init__(*args, **kwargs)
        self._data_changed = False

    def _process_inventory_lines(self, inv_lines):
        if False:
            print('Hello World!')
        'Generate a text key reference map rather for reconciling with.'
        repo = self._pack_collection.repo
        refs = repo._serializer._find_text_key_references(inv_lines)
        self._text_refs = refs
        self._text_filter = None

    def _copy_text_texts(self):
        if False:
            print('Hello World!')
        'generate what texts we should have and then copy.'
        self.pb.update('Copying content texts', 3)
        repo = self._pack_collection.repo
        ancestors = dict([(key[0], tuple((ref[0] for ref in refs[0]))) for (_1, key, _2, refs) in self.new_pack.revision_index.iter_all_entries()])
        ideal_index = repo._generate_text_key_index(self._text_refs, ancestors)
        ok_nodes = []
        bad_texts = []
        discarded_nodes = []
        NULL_REVISION = _mod_revision.NULL_REVISION
        (text_index_map, text_nodes) = self._get_text_nodes()
        for node in text_nodes:
            try:
                ideal_parents = tuple(ideal_index[node[1]])
            except KeyError:
                discarded_nodes.append(node)
                self._data_changed = True
            else:
                if ideal_parents == (NULL_REVISION,):
                    ideal_parents = ()
                if ideal_parents == node[3][0]:
                    ok_nodes.append(node)
                elif ideal_parents[0:1] == node[3][0][0:1]:
                    self._data_changed = True
                    ok_nodes.append((node[0], node[1], node[2], (ideal_parents, node[3][1])))
                    self._data_changed = True
                else:
                    bad_texts.append((node[1], ideal_parents))
                    self._data_changed = True
        del ideal_index
        del text_nodes
        (total_items, readv_group_iter) = self._least_readv_node_readv(ok_nodes)
        list(self._copy_nodes_graph(text_index_map, self.new_pack._writer, self.new_pack.text_index, readv_group_iter, total_items))
        topo_order = tsort.topo_sort(ancestors)
        rev_order = dict(zip(topo_order, range(len(topo_order))))
        bad_texts.sort(key=lambda key: rev_order.get(key[0][1], 0))
        transaction = repo.get_transaction()
        file_id_index = GraphIndexPrefixAdapter(self.new_pack.text_index, ('blank',), 1, add_nodes_callback=self.new_pack.text_index.add_nodes)
        data_access = _DirectPackAccess({self.new_pack.text_index: self.new_pack.access_tuple()})
        data_access.set_writer(self.new_pack._writer, self.new_pack.text_index, self.new_pack.access_tuple())
        output_texts = KnitVersionedFiles(_KnitGraphIndex(self.new_pack.text_index, add_callback=self.new_pack.text_index.add_nodes, deltas=True, parents=True, is_locked=repo.is_locked), data_access=data_access, max_delta_chain=200)
        for (key, parent_keys) in bad_texts:
            self.new_pack.flush()
            parents = []
            for parent_key in parent_keys:
                if parent_key[0] != key[0]:
                    raise errors.BzrError('Mismatched key parent %r:%r' % (key, parent_keys))
                parents.append(parent_key[1])
            text_lines = osutils.split_lines(repo.texts.get_record_stream([key], 'unordered', True).next().get_bytes_as('fulltext'))
            output_texts.add_lines(key, parent_keys, text_lines, random_id=True, check_content=False)
        missing_text_keys = self.new_pack.text_index._external_references()
        if missing_text_keys:
            raise errors.BzrCheckError('Reference to missing compression parents %r' % (missing_text_keys,))
        self._log_copied_texts()

    def _use_pack(self, new_pack):
        if False:
            return 10
        'Override _use_pack to check for reconcile having changed content.'
        original_inventory_keys = set()
        inv_index = self._pack_collection.inventory_index.combined_index
        for entry in inv_index.iter_all_entries():
            original_inventory_keys.add(entry[1])
        new_inventory_keys = set()
        for entry in new_pack.inventory_index.iter_all_entries():
            new_inventory_keys.add(entry[1])
        if new_inventory_keys != original_inventory_keys:
            self._data_changed = True
        return new_pack.data_inserted() and self._data_changed

class OptimisingKnitPacker(KnitPacker):
    """A packer which spends more time to create better disk layouts."""

    def _revision_node_readv(self, revision_nodes):
        if False:
            print('Hello World!')
        "Return the total revisions and the readv's to issue.\n\n        This sort places revisions in topological order with the ancestors\n        after the children.\n\n        :param revision_nodes: The revision index contents for the packs being\n            incorporated into the new pack.\n        :return: As per _least_readv_node_readv.\n        "
        ancestors = {}
        by_key = {}
        for (index, key, value, references) in revision_nodes:
            ancestors[key] = references[0]
            by_key[key] = (index, value, references)
        order = tsort.topo_sort(ancestors)
        total = len(order)
        requests = []
        for key in reversed(order):
            (index, value, references) = by_key[key]
            bits = value[1:].split(' ')
            (offset, length) = (int(bits[0]), int(bits[1]))
            requests.append((index, [(offset, length)], [(key, value[0], references)]))
        return (total, requests)

    def open_pack(self):
        if False:
            print('Hello World!')
        'Open a pack for the pack we are creating.'
        new_pack = super(OptimisingKnitPacker, self).open_pack()
        new_pack.revision_index.set_optimize(for_size=True)
        new_pack.inventory_index.set_optimize(for_size=True)
        new_pack.text_index.set_optimize(for_size=True)
        new_pack.signature_index.set_optimize(for_size=True)
        return new_pack

class KnitRepositoryPackCollection(RepositoryPackCollection):
    """A knit pack collection."""
    pack_factory = NewPack
    resumed_pack_factory = ResumedPack
    normal_packer_class = KnitPacker
    optimising_packer_class = OptimisingKnitPacker