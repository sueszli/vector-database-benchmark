from __future__ import absolute_import
from bzrlib.lazy_import import lazy_import
lazy_import(globals(), '\nimport itertools\n\nfrom bzrlib import (\n    controldir,\n    errors,\n    knit as _mod_knit,\n    lockable_files,\n    lockdir,\n    osutils,\n    revision as _mod_revision,\n    trace,\n    transactions,\n    versionedfile,\n    xml5,\n    xml6,\n    xml7,\n    )\n')
from bzrlib.decorators import needs_read_lock, needs_write_lock
from bzrlib.repository import InterRepository, IsInWriteGroupError, RepositoryFormatMetaDir
from bzrlib.vf_repository import InterSameDataRepository, MetaDirVersionedFileRepository, MetaDirVersionedFileRepositoryFormat, VersionedFileCommitBuilder, VersionedFileRootCommitBuilder
from bzrlib import symbol_versioning

class _KnitParentsProvider(object):

    def __init__(self, knit):
        if False:
            while True:
                i = 10
        self._knit = knit

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'KnitParentsProvider(%r)' % self._knit

    def get_parent_map(self, keys):
        if False:
            return 10
        'See graph.StackedParentsProvider.get_parent_map'
        parent_map = {}
        for revision_id in keys:
            if revision_id is None:
                raise ValueError('get_parent_map(None) is not valid')
            if revision_id == _mod_revision.NULL_REVISION:
                parent_map[revision_id] = ()
            else:
                try:
                    parents = tuple(self._knit.get_parents_with_ghosts(revision_id))
                except errors.RevisionNotPresent:
                    continue
                else:
                    if len(parents) == 0:
                        parents = (_mod_revision.NULL_REVISION,)
                parent_map[revision_id] = parents
        return parent_map

class _KnitsParentsProvider(object):

    def __init__(self, knit, prefix=()):
        if False:
            for i in range(10):
                print('nop')
        'Create a parent provider for string keys mapped to tuple keys.'
        self._knit = knit
        self._prefix = prefix

    def __repr__(self):
        if False:
            return 10
        return 'KnitsParentsProvider(%r)' % self._knit

    def get_parent_map(self, keys):
        if False:
            print('Hello World!')
        'See graph.StackedParentsProvider.get_parent_map'
        parent_map = self._knit.get_parent_map([self._prefix + (key,) for key in keys])
        result = {}
        for (key, parents) in parent_map.items():
            revid = key[-1]
            if len(parents) == 0:
                parents = (_mod_revision.NULL_REVISION,)
            else:
                parents = tuple((parent[-1] for parent in parents))
            result[revid] = parents
        for revision_id in keys:
            if revision_id == _mod_revision.NULL_REVISION:
                result[revision_id] = ()
        return result

class KnitRepository(MetaDirVersionedFileRepository):
    """Knit format repository."""
    _commit_builder_class = None
    _serializer = None

    def __init__(self, _format, a_bzrdir, control_files, _commit_builder_class, _serializer):
        if False:
            return 10
        super(KnitRepository, self).__init__(_format, a_bzrdir, control_files)
        self._commit_builder_class = _commit_builder_class
        self._serializer = _serializer
        self._reconcile_fixes_text_parents = True

    @needs_read_lock
    def _all_revision_ids(self):
        if False:
            return 10
        'See Repository.all_revision_ids().'
        return [key[0] for key in self.revisions.keys()]

    def _activate_new_inventory(self):
        if False:
            print('Hello World!')
        'Put a replacement inventory.new into use as inventories.'
        t = self._transport
        t.copy('inventory.new.kndx', 'inventory.kndx')
        try:
            t.copy('inventory.new.knit', 'inventory.knit')
        except errors.NoSuchFile:
            t.delete('inventory.knit')
        t.delete('inventory.new.kndx')
        try:
            t.delete('inventory.new.knit')
        except errors.NoSuchFile:
            pass
        self.inventories._index._reset_cache()
        self.inventories.keys()

    def _backup_inventory(self):
        if False:
            i = 10
            return i + 15
        t = self._transport
        t.copy('inventory.kndx', 'inventory.backup.kndx')
        t.copy('inventory.knit', 'inventory.backup.knit')

    def _move_file_id(self, from_id, to_id):
        if False:
            while True:
                i = 10
        t = self._transport.clone('knits')
        from_rel_url = self.texts._index._mapper.map((from_id, None))
        to_rel_url = self.texts._index._mapper.map((to_id, None))
        for suffix in ('.knit', '.kndx'):
            t.rename(from_rel_url + suffix, to_rel_url + suffix)

    def _remove_file_id(self, file_id):
        if False:
            return 10
        t = self._transport.clone('knits')
        rel_url = self.texts._index._mapper.map((file_id, None))
        for suffix in ('.kndx', '.knit'):
            try:
                t.delete(rel_url + suffix)
            except errors.NoSuchFile:
                pass

    def _temp_inventories(self):
        if False:
            i = 10
            return i + 15
        result = self._format._get_inventories(self._transport, self, 'inventory.new')
        result.get_parent_map([('A',)])
        return result

    @needs_read_lock
    def get_revision(self, revision_id):
        if False:
            return 10
        'Return the Revision object for a named revision'
        revision_id = osutils.safe_revision_id(revision_id)
        return self.get_revision_reconcile(revision_id)

    def _refresh_data(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.is_locked():
            return
        if self.is_in_write_group():
            raise IsInWriteGroupError(self)
        self.control_files._finish_transaction()
        if self.is_write_locked():
            self.control_files._set_write_transaction()
        else:
            self.control_files._set_read_transaction()

    @needs_write_lock
    def reconcile(self, other=None, thorough=False):
        if False:
            for i in range(10):
                print('nop')
        'Reconcile this repository.'
        from bzrlib.reconcile import KnitReconciler
        reconciler = KnitReconciler(self, thorough=thorough)
        reconciler.reconcile()
        return reconciler

    def _make_parents_provider(self):
        if False:
            while True:
                i = 10
        return _KnitsParentsProvider(self.revisions)

class RepositoryFormatKnit(MetaDirVersionedFileRepositoryFormat):
    """Bzr repository knit format (generalized).

    This repository format has:
     - knits for file texts and inventory
     - hash subdirectory based stores.
     - knits for revisions and signatures
     - TextStores for revisions and signatures.
     - a format marker of its own
     - an optional 'shared-storage' flag
     - an optional 'no-working-trees' flag
     - a LockDir lock
    """
    repository_class = None
    _commit_builder_class = None

    @property
    def _serializer(self):
        if False:
            for i in range(10):
                print('nop')
        return xml5.serializer_v5
    supports_ghosts = True
    supports_external_lookups = False
    supports_chks = False
    _fetch_order = 'topological'
    _fetch_uses_deltas = True
    fast_deltas = False
    supports_funky_characters = True
    revision_graph_can_have_wrong_parents = True

    def _get_inventories(self, repo_transport, repo, name='inventory'):
        if False:
            while True:
                i = 10
        mapper = versionedfile.ConstantMapper(name)
        index = _mod_knit._KndxIndex(repo_transport, mapper, repo.get_transaction, repo.is_write_locked, repo.is_locked)
        access = _mod_knit._KnitKeyAccess(repo_transport, mapper)
        return _mod_knit.KnitVersionedFiles(index, access, annotated=False)

    def _get_revisions(self, repo_transport, repo):
        if False:
            for i in range(10):
                print('nop')
        mapper = versionedfile.ConstantMapper('revisions')
        index = _mod_knit._KndxIndex(repo_transport, mapper, repo.get_transaction, repo.is_write_locked, repo.is_locked)
        access = _mod_knit._KnitKeyAccess(repo_transport, mapper)
        return _mod_knit.KnitVersionedFiles(index, access, max_delta_chain=0, annotated=False)

    def _get_signatures(self, repo_transport, repo):
        if False:
            print('Hello World!')
        mapper = versionedfile.ConstantMapper('signatures')
        index = _mod_knit._KndxIndex(repo_transport, mapper, repo.get_transaction, repo.is_write_locked, repo.is_locked)
        access = _mod_knit._KnitKeyAccess(repo_transport, mapper)
        return _mod_knit.KnitVersionedFiles(index, access, max_delta_chain=0, annotated=False)

    def _get_texts(self, repo_transport, repo):
        if False:
            i = 10
            return i + 15
        mapper = versionedfile.HashEscapedPrefixMapper()
        base_transport = repo_transport.clone('knits')
        index = _mod_knit._KndxIndex(base_transport, mapper, repo.get_transaction, repo.is_write_locked, repo.is_locked)
        access = _mod_knit._KnitKeyAccess(base_transport, mapper)
        return _mod_knit.KnitVersionedFiles(index, access, max_delta_chain=200, annotated=True)

    def initialize(self, a_bzrdir, shared=False):
        if False:
            return 10
        'Create a knit format 1 repository.\n\n        :param a_bzrdir: bzrdir to contain the new repository; must already\n            be initialized.\n        :param shared: If true the repository will be initialized as a shared\n                       repository.\n        '
        trace.mutter('creating repository in %s.', a_bzrdir.transport.base)
        dirs = ['knits']
        files = []
        utf8_files = [('format', self.get_format_string())]
        self._upload_blank_content(a_bzrdir, dirs, files, utf8_files, shared)
        repo_transport = a_bzrdir.get_repository_transport(None)
        control_files = lockable_files.LockableFiles(repo_transport, 'lock', lockdir.LockDir)
        transaction = transactions.WriteTransaction()
        result = self.open(a_bzrdir=a_bzrdir, _found=True)
        result.lock_write()
        result.inventories.get_parent_map([('A',)])
        result.revisions.get_parent_map([('A',)])
        result.signatures.get_parent_map([('A',)])
        result.unlock()
        self._run_post_repo_init_hooks(result, a_bzrdir, shared)
        return result

    def open(self, a_bzrdir, _found=False, _override_transport=None):
        if False:
            while True:
                i = 10
        "See RepositoryFormat.open().\n\n        :param _override_transport: INTERNAL USE ONLY. Allows opening the\n                                    repository at a slightly different url\n                                    than normal. I.e. during 'upgrade'.\n        "
        if not _found:
            format = RepositoryFormatMetaDir.find_format(a_bzrdir)
        if _override_transport is not None:
            repo_transport = _override_transport
        else:
            repo_transport = a_bzrdir.get_repository_transport(None)
        control_files = lockable_files.LockableFiles(repo_transport, 'lock', lockdir.LockDir)
        repo = self.repository_class(_format=self, a_bzrdir=a_bzrdir, control_files=control_files, _commit_builder_class=self._commit_builder_class, _serializer=self._serializer)
        repo.revisions = self._get_revisions(repo_transport, repo)
        repo.signatures = self._get_signatures(repo_transport, repo)
        repo.inventories = self._get_inventories(repo_transport, repo)
        repo.texts = self._get_texts(repo_transport, repo)
        repo.chk_bytes = None
        repo._transport = repo_transport
        return repo

class RepositoryFormatKnit1(RepositoryFormatKnit):
    """Bzr repository knit format 1.

    This repository format has:
     - knits for file texts and inventory
     - hash subdirectory based stores.
     - knits for revisions and signatures
     - TextStores for revisions and signatures.
     - a format marker of its own
     - an optional 'shared-storage' flag
     - an optional 'no-working-trees' flag
     - a LockDir lock

    This format was introduced in bzr 0.8.
    """
    repository_class = KnitRepository
    _commit_builder_class = VersionedFileCommitBuilder

    @property
    def _serializer(self):
        if False:
            while True:
                i = 10
        return xml5.serializer_v5

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return self.__class__ is not other.__class__

    @classmethod
    def get_format_string(cls):
        if False:
            return 10
        'See RepositoryFormat.get_format_string().'
        return 'Bazaar-NG Knit Repository Format 1'

    def get_format_description(self):
        if False:
            for i in range(10):
                print('nop')
        'See RepositoryFormat.get_format_description().'
        return 'Knit repository format 1'

class RepositoryFormatKnit3(RepositoryFormatKnit):
    """Bzr repository knit format 3.

    This repository format has:
     - knits for file texts and inventory
     - hash subdirectory based stores.
     - knits for revisions and signatures
     - TextStores for revisions and signatures.
     - a format marker of its own
     - an optional 'shared-storage' flag
     - an optional 'no-working-trees' flag
     - a LockDir lock
     - support for recording full info about the tree root
     - support for recording tree-references
    """
    repository_class = KnitRepository
    _commit_builder_class = VersionedFileRootCommitBuilder
    rich_root_data = True
    experimental = True
    supports_tree_reference = True

    @property
    def _serializer(self):
        if False:
            return 10
        return xml7.serializer_v7

    def _get_matching_bzrdir(self):
        if False:
            while True:
                i = 10
        return controldir.format_registry.make_bzrdir('dirstate-with-subtree')

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
        return 'Bazaar Knit Repository Format 3 (bzr 0.15)\n'

    def get_format_description(self):
        if False:
            return 10
        'See RepositoryFormat.get_format_description().'
        return 'Knit repository format 3'

class RepositoryFormatKnit4(RepositoryFormatKnit):
    """Bzr repository knit format 4.

    This repository format has everything in format 3, except for
    tree-references:
     - knits for file texts and inventory
     - hash subdirectory based stores.
     - knits for revisions and signatures
     - TextStores for revisions and signatures.
     - a format marker of its own
     - an optional 'shared-storage' flag
     - an optional 'no-working-trees' flag
     - a LockDir lock
     - support for recording full info about the tree root
    """
    repository_class = KnitRepository
    _commit_builder_class = VersionedFileRootCommitBuilder
    rich_root_data = True
    supports_tree_reference = False

    @property
    def _serializer(self):
        if False:
            return 10
        return xml6.serializer_v6

    def _get_matching_bzrdir(self):
        if False:
            i = 10
            return i + 15
        return controldir.format_registry.make_bzrdir('rich-root')

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
        return 'Bazaar Knit Repository Format 4 (bzr 1.0)\n'

    def get_format_description(self):
        if False:
            i = 10
            return i + 15
        'See RepositoryFormat.get_format_description().'
        return 'Knit repository format 4'

class InterKnitRepo(InterSameDataRepository):
    """Optimised code paths between Knit based repositories."""

    @classmethod
    def _get_repo_format_to_test(self):
        if False:
            return 10
        return RepositoryFormatKnit1()

    @staticmethod
    def is_compatible(source, target):
        if False:
            for i in range(10):
                print('nop')
        "Be compatible with known Knit formats.\n\n        We don't test for the stores being of specific types because that\n        could lead to confusing results, and there is no need to be\n        overly general.\n        "
        try:
            are_knits = isinstance(source._format, RepositoryFormatKnit) and isinstance(target._format, RepositoryFormatKnit)
        except AttributeError:
            return False
        return are_knits and InterRepository._same_model(source, target)

    @needs_read_lock
    def search_missing_revision_ids(self, find_ghosts=True, revision_ids=None, if_present_ids=None, limit=None):
        if False:
            print('Hello World!')
        'See InterRepository.search_missing_revision_ids().'
        source_ids_set = self._present_source_revisions_for(revision_ids, if_present_ids)
        target_ids = set(self.target.all_revision_ids())
        possibly_present_revisions = target_ids.intersection(source_ids_set)
        actually_present_revisions = set(self.target._eliminate_revisions_not_present(possibly_present_revisions))
        required_revisions = source_ids_set.difference(actually_present_revisions)
        if revision_ids is not None:
            result_set = required_revisions
        else:
            result_set = set(self.source._eliminate_revisions_not_present(required_revisions))
        if limit is not None:
            topo_ordered = self.source.get_graph().iter_topo_order(result_set)
            result_set = set(itertools.islice(topo_ordered, limit))
        return self.source.revision_ids_to_search_result(result_set)
InterRepository.register_optimiser(InterKnitRepo)