"""Reconcilers are able to fix some potential data errors in a branch."""
from __future__ import absolute_import
__all__ = ['KnitReconciler', 'PackReconciler', 'reconcile', 'Reconciler', 'RepoReconciler']
from bzrlib import cleanup, errors, revision as _mod_revision, ui
from bzrlib.trace import mutter
from bzrlib.tsort import topo_sort
from bzrlib.versionedfile import AdapterFactory, FulltextContentFactory
from bzrlib.i18n import gettext

def reconcile(dir, canonicalize_chks=False):
    if False:
        return 10
    "Reconcile the data in dir.\n\n    Currently this is limited to a inventory 'reweave'.\n\n    This is a convenience method, for using a Reconciler object.\n\n    Directly using Reconciler is recommended for library users that\n    desire fine grained control or analysis of the found issues.\n\n    :param canonicalize_chks: Make sure CHKs are in canonical form.\n    "
    reconciler = Reconciler(dir, canonicalize_chks=canonicalize_chks)
    reconciler.reconcile()

class Reconciler(object):
    """Reconcilers are used to reconcile existing data."""

    def __init__(self, dir, other=None, canonicalize_chks=False):
        if False:
            i = 10
            return i + 15
        'Create a Reconciler.'
        self.bzrdir = dir
        self.canonicalize_chks = canonicalize_chks

    def reconcile(self):
        if False:
            for i in range(10):
                print('nop')
        'Perform reconciliation.\n\n        After reconciliation the following attributes document found issues:\n\n        * `inconsistent_parents`: The number of revisions in the repository\n          whose ancestry was being reported incorrectly.\n        * `garbage_inventories`: The number of inventory objects without\n          revisions that were garbage collected.\n        * `fixed_branch_history`: None if there was no branch, False if the\n          branch history was correct, True if the branch history needed to be\n          re-normalized.\n        '
        self.pb = ui.ui_factory.nested_progress_bar()
        try:
            self._reconcile()
        finally:
            self.pb.finished()

    def _reconcile(self):
        if False:
            while True:
                i = 10
        'Helper function for performing reconciliation.'
        self._reconcile_branch()
        self._reconcile_repository()

    def _reconcile_branch(self):
        if False:
            return 10
        try:
            self.branch = self.bzrdir.open_branch()
        except errors.NotBranchError:
            self.fixed_branch_history = None
            return
        ui.ui_factory.note(gettext('Reconciling branch %s') % self.branch.base)
        branch_reconciler = self.branch.reconcile(thorough=True)
        self.fixed_branch_history = branch_reconciler.fixed_history

    def _reconcile_repository(self):
        if False:
            for i in range(10):
                print('nop')
        self.repo = self.bzrdir.find_repository()
        ui.ui_factory.note(gettext('Reconciling repository %s') % self.repo.user_url)
        self.pb.update(gettext('Reconciling repository'), 0, 1)
        if self.canonicalize_chks:
            try:
                self.repo.reconcile_canonicalize_chks
            except AttributeError:
                raise errors.BzrError(gettext('%s cannot canonicalize CHKs.') % (self.repo,))
            repo_reconciler = self.repo.reconcile_canonicalize_chks()
        else:
            repo_reconciler = self.repo.reconcile(thorough=True)
        self.inconsistent_parents = repo_reconciler.inconsistent_parents
        self.garbage_inventories = repo_reconciler.garbage_inventories
        if repo_reconciler.aborted:
            ui.ui_factory.note(gettext('Reconcile aborted: revision index has inconsistent parents.'))
            ui.ui_factory.note(gettext('Run "bzr check" for more details.'))
        else:
            ui.ui_factory.note(gettext('Reconciliation complete.'))

class BranchReconciler(object):
    """Reconciler that works on a branch."""

    def __init__(self, a_branch, thorough=False):
        if False:
            i = 10
            return i + 15
        self.fixed_history = None
        self.thorough = thorough
        self.branch = a_branch

    def reconcile(self):
        if False:
            for i in range(10):
                print('nop')
        operation = cleanup.OperationWithCleanups(self._reconcile)
        self.add_cleanup = operation.add_cleanup
        operation.run_simple()

    def _reconcile(self):
        if False:
            return 10
        self.branch.lock_write()
        self.add_cleanup(self.branch.unlock)
        self.pb = ui.ui_factory.nested_progress_bar()
        self.add_cleanup(self.pb.finished)
        self._reconcile_steps()

    def _reconcile_steps(self):
        if False:
            i = 10
            return i + 15
        self._reconcile_revision_history()

    def _reconcile_revision_history(self):
        if False:
            return 10
        (last_revno, last_revision_id) = self.branch.last_revision_info()
        real_history = []
        graph = self.branch.repository.get_graph()
        try:
            for revid in graph.iter_lefthand_ancestry(last_revision_id, (_mod_revision.NULL_REVISION,)):
                real_history.append(revid)
        except errors.RevisionNotPresent:
            pass
        real_history.reverse()
        if last_revno != len(real_history):
            self.fixed_history = True
            ui.ui_factory.note(gettext('Fixing last revision info {0}  => {1}').format(last_revno, len(real_history)))
            self.branch.set_last_revision_info(len(real_history), last_revision_id)
        else:
            self.fixed_history = False
            ui.ui_factory.note(gettext('revision_history ok.'))

class RepoReconciler(object):
    """Reconciler that reconciles a repository.

    The goal of repository reconciliation is to make any derived data
    consistent with the core data committed by a user. This can involve
    reindexing, or removing unreferenced data if that can interfere with
    queries in a given repository.

    Currently this consists of an inventory reweave with revision cross-checks.
    """

    def __init__(self, repo, other=None, thorough=False):
        if False:
            for i in range(10):
                print('nop')
        'Construct a RepoReconciler.\n\n        :param thorough: perform a thorough check which may take longer but\n                         will correct non-data loss issues such as incorrect\n                         cached data.\n        '
        self.garbage_inventories = 0
        self.inconsistent_parents = 0
        self.aborted = False
        self.repo = repo
        self.thorough = thorough

    def reconcile(self):
        if False:
            print('Hello World!')
        'Perform reconciliation.\n\n        After reconciliation the following attributes document found issues:\n\n        * `inconsistent_parents`: The number of revisions in the repository\n          whose ancestry was being reported incorrectly.\n        * `garbage_inventories`: The number of inventory objects without\n          revisions that were garbage collected.\n        '
        operation = cleanup.OperationWithCleanups(self._reconcile)
        self.add_cleanup = operation.add_cleanup
        operation.run_simple()

    def _reconcile(self):
        if False:
            while True:
                i = 10
        self.repo.lock_write()
        self.add_cleanup(self.repo.unlock)
        self.pb = ui.ui_factory.nested_progress_bar()
        self.add_cleanup(self.pb.finished)
        self._reconcile_steps()

    def _reconcile_steps(self):
        if False:
            return 10
        'Perform the steps to reconcile this repository.'
        self._reweave_inventory()

    def _reweave_inventory(self):
        if False:
            print('Hello World!')
        'Regenerate the inventory weave for the repository from scratch.\n\n        This is a smart function: it will only do the reweave if doing it\n        will correct data issues. The self.thorough flag controls whether\n        only data-loss causing issues (!self.thorough) or all issues\n        (self.thorough) are treated as requiring the reweave.\n        '
        transaction = self.repo.get_transaction()
        self.pb.update(gettext('Reading inventory data'))
        self.inventory = self.repo.inventories
        self.revisions = self.repo.revisions
        self.pending = set([key[-1] for key in self.revisions.keys()])
        self._rev_graph = {}
        self.inconsistent_parents = 0
        self._setup_steps(len(self.pending))
        for rev_id in self.pending:
            self._graph_revision(rev_id)
        self._check_garbage_inventories()
        if not self.inconsistent_parents and (not self.garbage_inventories or not self.thorough):
            ui.ui_factory.note(gettext('Inventory ok.'))
            return
        self.pb.update(gettext('Backing up inventory'), 0, 0)
        self.repo._backup_inventory()
        ui.ui_factory.note(gettext('Backup inventory created.'))
        new_inventories = self.repo._temp_inventories()
        self._setup_steps(len(self._rev_graph))
        revision_keys = [(rev_id,) for rev_id in topo_sort(self._rev_graph)]
        stream = self._change_inv_parents(self.inventory.get_record_stream(revision_keys, 'unordered', True), self._new_inv_parents, set(revision_keys))
        new_inventories.insert_record_stream(stream)
        if not set(new_inventories.keys()) == set([(revid,) for revid in self.pending]):
            raise AssertionError()
        self.pb.update(gettext('Writing weave'))
        self.repo._activate_new_inventory()
        self.inventory = None
        ui.ui_factory.note(gettext('Inventory regenerated.'))

    def _new_inv_parents(self, revision_key):
        if False:
            print('Hello World!')
        'Lookup ghost-filtered parents for revision_key.'
        return tuple([(revid,) for revid in self._rev_graph[revision_key[-1]]])

    def _change_inv_parents(self, stream, get_parents, all_revision_keys):
        if False:
            print('Hello World!')
        'Adapt a record stream to reconcile the parents.'
        for record in stream:
            wanted_parents = get_parents(record.key)
            if wanted_parents and wanted_parents[0] not in all_revision_keys:
                bytes = record.get_bytes_as('fulltext')
                yield FulltextContentFactory(record.key, wanted_parents, record.sha1, bytes)
            else:
                adapted_record = AdapterFactory(record.key, wanted_parents, record)
                yield adapted_record
            self._reweave_step('adding inventories')

    def _setup_steps(self, new_total):
        if False:
            for i in range(10):
                print('nop')
        'Setup the markers we need to control the progress bar.'
        self.total = new_total
        self.count = 0

    def _graph_revision(self, rev_id):
        if False:
            print('Hello World!')
        'Load a revision into the revision graph.'
        self._reweave_step('loading revisions')
        rev = self.repo.get_revision_reconcile(rev_id)
        parents = []
        for parent in rev.parent_ids:
            if self._parent_is_available(parent):
                parents.append(parent)
            else:
                mutter('found ghost %s', parent)
        self._rev_graph[rev_id] = parents

    def _check_garbage_inventories(self):
        if False:
            for i in range(10):
                print('nop')
        'Check for garbage inventories which we cannot trust\n\n        We cant trust them because their pre-requisite file data may not\n        be present - all we know is that their revision was not installed.\n        '
        if not self.thorough:
            return
        inventories = set(self.inventory.keys())
        revisions = set(self.revisions.keys())
        garbage = inventories.difference(revisions)
        self.garbage_inventories = len(garbage)
        for revision_key in garbage:
            mutter('Garbage inventory {%s} found.', revision_key[-1])

    def _parent_is_available(self, parent):
        if False:
            return 10
        'True if parent is a fully available revision\n\n        A fully available revision has a inventory and a revision object in the\n        repository.\n        '
        if parent in self._rev_graph:
            return True
        inv_present = 1 == len(self.inventory.get_parent_map([(parent,)]))
        return inv_present and self.repo.has_revision(parent)

    def _reweave_step(self, message):
        if False:
            print('Hello World!')
        'Mark a single step of regeneration complete.'
        self.pb.update(message, self.count, self.total)
        self.count += 1

class KnitReconciler(RepoReconciler):
    """Reconciler that reconciles a knit format repository.

    This will detect garbage inventories and remove them in thorough mode.
    """

    def _reconcile_steps(self):
        if False:
            for i in range(10):
                print('nop')
        'Perform the steps to reconcile this repository.'
        if self.thorough:
            try:
                self._load_indexes()
            except errors.BzrCheckError:
                self.aborted = True
                return
            self._gc_inventory()
            self._fix_text_parents()

    def _load_indexes(self):
        if False:
            return 10
        'Load indexes for the reconciliation.'
        self.transaction = self.repo.get_transaction()
        self.pb.update(gettext('Reading indexes'), 0, 2)
        self.inventory = self.repo.inventories
        self.pb.update(gettext('Reading indexes'), 1, 2)
        self.repo._check_for_inconsistent_revision_parents()
        self.revisions = self.repo.revisions
        self.pb.update(gettext('Reading indexes'), 2, 2)

    def _gc_inventory(self):
        if False:
            for i in range(10):
                print('nop')
        'Remove inventories that are not referenced from the revision store.'
        self.pb.update(gettext('Checking unused inventories'), 0, 1)
        self._check_garbage_inventories()
        self.pb.update(gettext('Checking unused inventories'), 1, 3)
        if not self.garbage_inventories:
            ui.ui_factory.note(gettext('Inventory ok.'))
            return
        self.pb.update(gettext('Backing up inventory'), 0, 0)
        self.repo._backup_inventory()
        ui.ui_factory.note(gettext('Backup Inventory created'))
        new_inventories = self.repo._temp_inventories()
        graph = self.revisions.get_parent_map(self.revisions.keys())
        revision_keys = topo_sort(graph)
        revision_ids = [key[-1] for key in revision_keys]
        self._setup_steps(len(revision_keys))
        stream = self._change_inv_parents(self.inventory.get_record_stream(revision_keys, 'unordered', True), graph.__getitem__, set(revision_keys))
        new_inventories.insert_record_stream(stream)
        if not set(new_inventories.keys()) == set(revision_keys):
            raise AssertionError()
        self.pb.update(gettext('Writing weave'))
        self.repo._activate_new_inventory()
        self.inventory = None
        ui.ui_factory.note(gettext('Inventory regenerated.'))

    def _fix_text_parents(self):
        if False:
            return 10
        'Fix bad versionedfile parent entries.\n\n        It is possible for the parents entry in a versionedfile entry to be\n        inconsistent with the values in the revision and inventory.\n\n        This method finds entries with such inconsistencies, corrects their\n        parent lists, and replaces the versionedfile with a corrected version.\n        '
        transaction = self.repo.get_transaction()
        versions = [key[-1] for key in self.revisions.keys()]
        mutter('Prepopulating revision text cache with %d revisions', len(versions))
        vf_checker = self.repo._get_versioned_file_checker()
        (bad_parents, unused_versions) = vf_checker.check_file_version_parents(self.repo.texts, self.pb)
        text_index = vf_checker.text_index
        per_id_bad_parents = {}
        for key in unused_versions:
            per_id_bad_parents[key[0]] = {}
        for (key, details) in bad_parents.iteritems():
            file_id = key[0]
            rev_id = key[1]
            knit_parents = tuple([parent[-1] for parent in details[0]])
            correct_parents = tuple([parent[-1] for parent in details[1]])
            file_details = per_id_bad_parents.setdefault(file_id, {})
            file_details[rev_id] = (knit_parents, correct_parents)
        file_id_versions = {}
        for text_key in text_index:
            versions_list = file_id_versions.setdefault(text_key[0], [])
            versions_list.append(text_key[1])
        for (num, file_id) in enumerate(per_id_bad_parents):
            self.pb.update(gettext('Fixing text parents'), num, len(per_id_bad_parents))
            versions_with_bad_parents = per_id_bad_parents[file_id]
            id_unused_versions = set((key[-1] for key in unused_versions if key[0] == file_id))
            if file_id in file_id_versions:
                file_versions = file_id_versions[file_id]
            else:
                file_versions = []
            self._fix_text_parent(file_id, versions_with_bad_parents, id_unused_versions, file_versions)

    def _fix_text_parent(self, file_id, versions_with_bad_parents, unused_versions, all_versions):
        if False:
            print('Hello World!')
        'Fix bad versionedfile entries in a single versioned file.'
        mutter('fixing text parent: %r (%d versions)', file_id, len(versions_with_bad_parents))
        mutter('(%d are unused)', len(unused_versions))
        new_file_id = 'temp:%s' % file_id
        new_parents = {}
        needed_keys = set()
        for version in all_versions:
            if version in unused_versions:
                continue
            elif version in versions_with_bad_parents:
                parents = versions_with_bad_parents[version][1]
            else:
                pmap = self.repo.texts.get_parent_map([(file_id, version)])
                parents = [key[-1] for key in pmap[file_id, version]]
            new_parents[new_file_id, version] = [(new_file_id, parent) for parent in parents]
            needed_keys.add((file_id, version))

        def fix_parents(stream):
            if False:
                i = 10
                return i + 15
            for record in stream:
                bytes = record.get_bytes_as('fulltext')
                new_key = (new_file_id, record.key[-1])
                parents = new_parents[new_key]
                yield FulltextContentFactory(new_key, parents, record.sha1, bytes)
        stream = self.repo.texts.get_record_stream(needed_keys, 'topological', True)
        self.repo._remove_file_id(new_file_id)
        self.repo.texts.insert_record_stream(fix_parents(stream))
        self.repo._remove_file_id(file_id)
        if len(new_parents):
            self.repo._move_file_id(new_file_id, file_id)

class PackReconciler(RepoReconciler):
    """Reconciler that reconciles a pack based repository.

    Garbage inventories do not affect ancestry queries, and removal is
    considerably more expensive as there is no separate versioned file for
    them, so they are not cleaned. In short it is currently a no-op.

    In future this may be a good place to hook in annotation cache checking,
    index recreation etc.
    """

    def __init__(self, repo, other=None, thorough=False, canonicalize_chks=False):
        if False:
            while True:
                i = 10
        super(PackReconciler, self).__init__(repo, other=other, thorough=thorough)
        self.canonicalize_chks = canonicalize_chks

    def _reconcile_steps(self):
        if False:
            print('Hello World!')
        'Perform the steps to reconcile this repository.'
        if not self.thorough:
            return
        collection = self.repo._pack_collection
        collection.ensure_loaded()
        collection.lock_names()
        self.add_cleanup(collection._unlock_names)
        packs = collection.all_packs()
        all_revisions = self.repo.all_revision_ids()
        total_inventories = len(list(collection.inventory_index.combined_index.iter_all_entries()))
        if len(all_revisions):
            if self.canonicalize_chks:
                reconcile_meth = self.repo._canonicalize_chks_pack
            else:
                reconcile_meth = self.repo._reconcile_pack
            new_pack = reconcile_meth(collection, packs, '.reconcile', all_revisions, self.pb)
            if new_pack is not None:
                self._discard_and_save(packs)
        else:
            self._discard_and_save(packs)
        self.garbage_inventories = total_inventories - len(list(collection.inventory_index.combined_index.iter_all_entries()))

    def _discard_and_save(self, packs):
        if False:
            while True:
                i = 10
        'Discard some packs from the repository.\n\n        This removes them from the memory index, saves the in-memory index\n        which makes the newly reconciled pack visible and hides the packs to be\n        discarded, and finally renames the packs being discarded into the\n        obsolete packs directory.\n\n        :param packs: The packs to discard.\n        '
        for pack in packs:
            self.repo._pack_collection._remove_pack_from_memory(pack)
        self.repo._pack_collection._save_pack_names()
        self.repo._pack_collection._obsolete_packs(packs)