"""Versioned text file storage api."""
from __future__ import absolute_import
from copy import copy
from cStringIO import StringIO
import os
import struct
from zlib import adler32
from bzrlib.lazy_import import lazy_import
lazy_import(globals(), '\nfrom bzrlib import (\n    annotate,\n    bencode,\n    errors,\n    graph as _mod_graph,\n    groupcompress,\n    index,\n    knit,\n    osutils,\n    multiparent,\n    tsort,\n    revision,\n    urlutils,\n    )\n')
from bzrlib.registry import Registry
from bzrlib.textmerge import TextMerge
adapter_registry = Registry()
adapter_registry.register_lazy(('knit-delta-gz', 'fulltext'), 'bzrlib.knit', 'DeltaPlainToFullText')
adapter_registry.register_lazy(('knit-ft-gz', 'fulltext'), 'bzrlib.knit', 'FTPlainToFullText')
adapter_registry.register_lazy(('knit-annotated-delta-gz', 'knit-delta-gz'), 'bzrlib.knit', 'DeltaAnnotatedToUnannotated')
adapter_registry.register_lazy(('knit-annotated-delta-gz', 'fulltext'), 'bzrlib.knit', 'DeltaAnnotatedToFullText')
adapter_registry.register_lazy(('knit-annotated-ft-gz', 'knit-ft-gz'), 'bzrlib.knit', 'FTAnnotatedToUnannotated')
adapter_registry.register_lazy(('knit-annotated-ft-gz', 'fulltext'), 'bzrlib.knit', 'FTAnnotatedToFullText')

class ContentFactory(object):
    """Abstract interface for insertion and retrieval from a VersionedFile.

    :ivar sha1: None, or the sha1 of the content fulltext.
    :ivar storage_kind: The native storage kind of this factory. One of
        'mpdiff', 'knit-annotated-ft', 'knit-annotated-delta', 'knit-ft',
        'knit-delta', 'fulltext', 'knit-annotated-ft-gz',
        'knit-annotated-delta-gz', 'knit-ft-gz', 'knit-delta-gz'.
    :ivar key: The key of this content. Each key is a tuple with a single
        string in it.
    :ivar parents: A tuple of parent keys for self.key. If the object has
        no parent information, None (as opposed to () for an empty list of
        parents).
    """

    def __init__(self):
        if False:
            print('Hello World!')
        'Create a ContentFactory.'
        self.sha1 = None
        self.storage_kind = None
        self.key = None
        self.parents = None

class ChunkedContentFactory(ContentFactory):
    """Static data content factory.

    This takes a 'chunked' list of strings. The only requirement on 'chunked' is
    that ''.join(lines) becomes a valid fulltext. A tuple of a single string
    satisfies this, as does a list of lines.

    :ivar sha1: None, or the sha1 of the content fulltext.
    :ivar storage_kind: The native storage kind of this factory. Always
        'chunked'
    :ivar key: The key of this content. Each key is a tuple with a single
        string in it.
    :ivar parents: A tuple of parent keys for self.key. If the object has
        no parent information, None (as opposed to () for an empty list of
        parents).
     """

    def __init__(self, key, parents, sha1, chunks):
        if False:
            print('Hello World!')
        'Create a ContentFactory.'
        self.sha1 = sha1
        self.storage_kind = 'chunked'
        self.key = key
        self.parents = parents
        self._chunks = chunks

    def get_bytes_as(self, storage_kind):
        if False:
            i = 10
            return i + 15
        if storage_kind == 'chunked':
            return self._chunks
        elif storage_kind == 'fulltext':
            return ''.join(self._chunks)
        raise errors.UnavailableRepresentation(self.key, storage_kind, self.storage_kind)

class FulltextContentFactory(ContentFactory):
    """Static data content factory.

    This takes a fulltext when created and just returns that during
    get_bytes_as('fulltext').

    :ivar sha1: None, or the sha1 of the content fulltext.
    :ivar storage_kind: The native storage kind of this factory. Always
        'fulltext'.
    :ivar key: The key of this content. Each key is a tuple with a single
        string in it.
    :ivar parents: A tuple of parent keys for self.key. If the object has
        no parent information, None (as opposed to () for an empty list of
        parents).
     """

    def __init__(self, key, parents, sha1, text):
        if False:
            i = 10
            return i + 15
        'Create a ContentFactory.'
        self.sha1 = sha1
        self.storage_kind = 'fulltext'
        self.key = key
        self.parents = parents
        self._text = text

    def get_bytes_as(self, storage_kind):
        if False:
            i = 10
            return i + 15
        if storage_kind == self.storage_kind:
            return self._text
        elif storage_kind == 'chunked':
            return [self._text]
        raise errors.UnavailableRepresentation(self.key, storage_kind, self.storage_kind)

class AbsentContentFactory(ContentFactory):
    """A placeholder content factory for unavailable texts.

    :ivar sha1: None.
    :ivar storage_kind: 'absent'.
    :ivar key: The key of this content. Each key is a tuple with a single
        string in it.
    :ivar parents: None.
    """

    def __init__(self, key):
        if False:
            print('Hello World!')
        'Create a ContentFactory.'
        self.sha1 = None
        self.storage_kind = 'absent'
        self.key = key
        self.parents = None

    def get_bytes_as(self, storage_kind):
        if False:
            print('Hello World!')
        raise ValueError('A request was made for key: %s, but that content is not available, and the calling code does not handle if it is missing.' % (self.key,))

class AdapterFactory(ContentFactory):
    """A content factory to adapt between key prefix's."""

    def __init__(self, key, parents, adapted):
        if False:
            return 10
        'Create an adapter factory instance.'
        self.key = key
        self.parents = parents
        self._adapted = adapted

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        'Return a member from the adapted object.'
        if attr in ('key', 'parents'):
            return self.__dict__[attr]
        else:
            return getattr(self._adapted, attr)

def filter_absent(record_stream):
    if False:
        print('Hello World!')
    'Adapt a record stream to remove absent records.'
    for record in record_stream:
        if record.storage_kind != 'absent':
            yield record

class _MPDiffGenerator(object):
    """Pull out the functionality for generating mp_diffs."""

    def __init__(self, vf, keys):
        if False:
            while True:
                i = 10
        self.vf = vf
        self.ordered_keys = tuple(keys)
        self.needed_keys = ()
        self.diffs = {}
        self.parent_map = {}
        self.ghost_parents = ()
        self.refcounts = {}
        self.chunks = {}

    def _find_needed_keys(self):
        if False:
            return 10
        'Find the set of keys we need to request.\n\n        This includes all the original keys passed in, and the non-ghost\n        parents of those keys.\n\n        :return: (needed_keys, refcounts)\n            needed_keys is the set of all texts we need to extract\n            refcounts is a dict of {key: num_children} letting us know when we\n                no longer need to cache a given parent text\n        '
        needed_keys = set(self.ordered_keys)
        parent_map = self.vf.get_parent_map(needed_keys)
        self.parent_map = parent_map
        missing_keys = needed_keys.difference(parent_map)
        if missing_keys:
            raise errors.RevisionNotPresent(list(missing_keys)[0], self.vf)
        refcounts = {}
        setdefault = refcounts.setdefault
        just_parents = set()
        for (child_key, parent_keys) in parent_map.iteritems():
            if not parent_keys:
                continue
            just_parents.update(parent_keys)
            needed_keys.update(parent_keys)
            for p in parent_keys:
                refcounts[p] = setdefault(p, 0) + 1
        just_parents.difference_update(parent_map)
        self.present_parents = set(self.vf.get_parent_map(just_parents))
        self.ghost_parents = just_parents.difference(self.present_parents)
        needed_keys.difference_update(self.ghost_parents)
        self.needed_keys = needed_keys
        self.refcounts = refcounts
        return (needed_keys, refcounts)

    def _compute_diff(self, key, parent_lines, lines):
        if False:
            i = 10
            return i + 15
        'Compute a single mp_diff, and store it in self._diffs'
        if len(parent_lines) > 0:
            left_parent_blocks = self.vf._extract_blocks(key, parent_lines[0], lines)
        else:
            left_parent_blocks = None
        diff = multiparent.MultiParent.from_lines(lines, parent_lines, left_parent_blocks)
        self.diffs[key] = diff

    def _process_one_record(self, key, this_chunks):
        if False:
            while True:
                i = 10
        parent_keys = None
        if key in self.parent_map:
            parent_keys = self.parent_map.pop(key)
            if parent_keys is None:
                parent_keys = ()
            parent_lines = []
            for p in parent_keys:
                if p in self.ghost_parents:
                    continue
                refcount = self.refcounts[p]
                if refcount == 1:
                    self.refcounts.pop(p)
                    parent_chunks = self.chunks.pop(p)
                else:
                    self.refcounts[p] = refcount - 1
                    parent_chunks = self.chunks[p]
                p_lines = osutils.chunks_to_lines(parent_chunks)
                parent_lines.append(p_lines)
                del p_lines
            lines = osutils.chunks_to_lines(this_chunks)
            this_chunks = lines
            self._compute_diff(key, parent_lines, lines)
            del lines
        if key in self.refcounts:
            self.chunks[key] = this_chunks

    def _extract_diffs(self):
        if False:
            print('Hello World!')
        (needed_keys, refcounts) = self._find_needed_keys()
        for record in self.vf.get_record_stream(needed_keys, 'topological', True):
            if record.storage_kind == 'absent':
                raise errors.RevisionNotPresent(record.key, self.vf)
            self._process_one_record(record.key, record.get_bytes_as('chunked'))

    def compute_diffs(self):
        if False:
            print('Hello World!')
        self._extract_diffs()
        dpop = self.diffs.pop
        return [dpop(k) for k in self.ordered_keys]

class VersionedFile(object):
    """Versioned text file storage.

    A versioned file manages versions of line-based text files,
    keeping track of the originating version for each line.

    To clients the "lines" of the file are represented as a list of
    strings. These strings will typically have terminal newline
    characters, but this is not required.  In particular files commonly
    do not have a newline at the end of the file.

    Texts are identified by a version-id string.
    """

    @staticmethod
    def check_not_reserved_id(version_id):
        if False:
            for i in range(10):
                print('nop')
        revision.check_not_reserved_id(version_id)

    def copy_to(self, name, transport):
        if False:
            while True:
                i = 10
        'Copy this versioned file to name on transport.'
        raise NotImplementedError(self.copy_to)

    def get_record_stream(self, versions, ordering, include_delta_closure):
        if False:
            print('Hello World!')
        "Get a stream of records for versions.\n\n        :param versions: The versions to include. Each version is a tuple\n            (version,).\n        :param ordering: Either 'unordered' or 'topological'. A topologically\n            sorted stream has compression parents strictly before their\n            children.\n        :param include_delta_closure: If True then the closure across any\n            compression parents will be included (in the data content of the\n            stream, not in the emitted records). This guarantees that\n            'fulltext' can be used successfully on every record.\n        :return: An iterator of ContentFactory objects, each of which is only\n            valid until the iterator is advanced.\n        "
        raise NotImplementedError(self.get_record_stream)

    def has_version(self, version_id):
        if False:
            print('Hello World!')
        'Returns whether version is present.'
        raise NotImplementedError(self.has_version)

    def insert_record_stream(self, stream):
        if False:
            print('Hello World!')
        'Insert a record stream into this versioned file.\n\n        :param stream: A stream of records to insert.\n        :return: None\n        :seealso VersionedFile.get_record_stream:\n        '
        raise NotImplementedError

    def add_lines(self, version_id, parents, lines, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False, check_content=True):
        if False:
            print('Hello World!')
        "Add a single text on top of the versioned file.\n\n        Must raise RevisionAlreadyPresent if the new version is\n        already present in file history.\n\n        Must raise RevisionNotPresent if any of the given parents are\n        not present in file history.\n\n        :param lines: A list of lines. Each line must be a bytestring. And all\n            of them except the last must be terminated with \n and contain no\n            other \n's. The last line may either contain no \n's or a single\n            terminated \n. If the lines list does meet this constraint the add\n            routine may error or may succeed - but you will be unable to read\n            the data back accurately. (Checking the lines have been split\n            correctly is expensive and extremely unlikely to catch bugs so it\n            is not done at runtime unless check_content is True.)\n        :param parent_texts: An optional dictionary containing the opaque\n            representations of some or all of the parents of version_id to\n            allow delta optimisations.  VERY IMPORTANT: the texts must be those\n            returned by add_lines or data corruption can be caused.\n        :param left_matching_blocks: a hint about which areas are common\n            between the text and its left-hand-parent.  The format is\n            the SequenceMatcher.get_matching_blocks format.\n        :param nostore_sha: Raise ExistingContent and do not add the lines to\n            the versioned file if the digest of the lines matches this.\n        :param random_id: If True a random id has been selected rather than\n            an id determined by some deterministic process such as a converter\n            from a foreign VCS. When True the backend may choose not to check\n            for uniqueness of the resulting key within the versioned file, so\n            this should only be done when the result is expected to be unique\n            anyway.\n        :param check_content: If True, the lines supplied are verified to be\n            bytestrings that are correctly formed lines.\n        :return: The text sha1, the number of bytes in the text, and an opaque\n                 representation of the inserted version which can be provided\n                 back to future add_lines calls in the parent_texts dictionary.\n        "
        self._check_write_ok()
        return self._add_lines(version_id, parents, lines, parent_texts, left_matching_blocks, nostore_sha, random_id, check_content)

    def _add_lines(self, version_id, parents, lines, parent_texts, left_matching_blocks, nostore_sha, random_id, check_content):
        if False:
            i = 10
            return i + 15
        'Helper to do the class specific add_lines.'
        raise NotImplementedError(self.add_lines)

    def add_lines_with_ghosts(self, version_id, parents, lines, parent_texts=None, nostore_sha=None, random_id=False, check_content=True, left_matching_blocks=None):
        if False:
            print('Hello World!')
        'Add lines to the versioned file, allowing ghosts to be present.\n\n        This takes the same parameters as add_lines and returns the same.\n        '
        self._check_write_ok()
        return self._add_lines_with_ghosts(version_id, parents, lines, parent_texts, nostore_sha, random_id, check_content, left_matching_blocks)

    def _add_lines_with_ghosts(self, version_id, parents, lines, parent_texts, nostore_sha, random_id, check_content, left_matching_blocks):
        if False:
            print('Hello World!')
        'Helper to do class specific add_lines_with_ghosts.'
        raise NotImplementedError(self.add_lines_with_ghosts)

    def check(self, progress_bar=None):
        if False:
            while True:
                i = 10
        'Check the versioned file for integrity.'
        raise NotImplementedError(self.check)

    def _check_lines_not_unicode(self, lines):
        if False:
            i = 10
            return i + 15
        'Check that lines being added to a versioned file are not unicode.'
        for line in lines:
            if line.__class__ is not str:
                raise errors.BzrBadParameterUnicode('lines')

    def _check_lines_are_lines(self, lines):
        if False:
            while True:
                i = 10
        'Check that the lines really are full lines without inline EOL.'
        for line in lines:
            if '\n' in line[:-1]:
                raise errors.BzrBadParameterContainsNewline('lines')

    def get_format_signature(self):
        if False:
            print('Hello World!')
        'Get a text description of the data encoding in this file.\n\n        :since: 0.90\n        '
        raise NotImplementedError(self.get_format_signature)

    def make_mpdiffs(self, version_ids):
        if False:
            for i in range(10):
                print('nop')
        'Create multiparent diffs for specified versions.'
        knit_versions = set()
        knit_versions.update(version_ids)
        parent_map = self.get_parent_map(version_ids)
        for version_id in version_ids:
            try:
                knit_versions.update(parent_map[version_id])
            except KeyError:
                raise errors.RevisionNotPresent(version_id, self)
        knit_versions = set(self.get_parent_map(knit_versions).keys())
        lines = dict(zip(knit_versions, self._get_lf_split_line_list(knit_versions)))
        diffs = []
        for version_id in version_ids:
            target = lines[version_id]
            try:
                parents = [lines[p] for p in parent_map[version_id] if p in knit_versions]
            except KeyError:
                raise errors.RevisionNotPresent(version_id, self)
            if len(parents) > 0:
                left_parent_blocks = self._extract_blocks(version_id, parents[0], target)
            else:
                left_parent_blocks = None
            diffs.append(multiparent.MultiParent.from_lines(target, parents, left_parent_blocks))
        return diffs

    def _extract_blocks(self, version_id, source, target):
        if False:
            i = 10
            return i + 15
        return None

    def add_mpdiffs(self, records):
        if False:
            i = 10
            return i + 15
        'Add mpdiffs to this VersionedFile.\n\n        Records should be iterables of version, parents, expected_sha1,\n        mpdiff. mpdiff should be a MultiParent instance.\n        '
        vf_parents = {}
        mpvf = multiparent.MultiMemoryVersionedFile()
        versions = []
        for (version, parent_ids, expected_sha1, mpdiff) in records:
            versions.append(version)
            mpvf.add_diff(mpdiff, version, parent_ids)
        needed_parents = set()
        for (version, parent_ids, expected_sha1, mpdiff) in records:
            needed_parents.update((p for p in parent_ids if not mpvf.has_version(p)))
        present_parents = set(self.get_parent_map(needed_parents).keys())
        for (parent_id, lines) in zip(present_parents, self._get_lf_split_line_list(present_parents)):
            mpvf.add_version(lines, parent_id, [])
        for ((version, parent_ids, expected_sha1, mpdiff), lines) in zip(records, mpvf.get_line_list(versions)):
            if len(parent_ids) == 1:
                left_matching_blocks = list(mpdiff.get_matching_blocks(0, mpvf.get_diff(parent_ids[0]).num_lines()))
            else:
                left_matching_blocks = None
            try:
                (_, _, version_text) = self.add_lines_with_ghosts(version, parent_ids, lines, vf_parents, left_matching_blocks=left_matching_blocks)
            except NotImplementedError:
                (_, _, version_text) = self.add_lines(version, parent_ids, lines, vf_parents, left_matching_blocks=left_matching_blocks)
            vf_parents[version] = version_text
        sha1s = self.get_sha1s(versions)
        for (version, parent_ids, expected_sha1, mpdiff) in records:
            if expected_sha1 != sha1s[version]:
                raise errors.VersionedFileInvalidChecksum(version)

    def get_text(self, version_id):
        if False:
            print('Hello World!')
        'Return version contents as a text string.\n\n        Raises RevisionNotPresent if version is not present in\n        file history.\n        '
        return ''.join(self.get_lines(version_id))
    get_string = get_text

    def get_texts(self, version_ids):
        if False:
            for i in range(10):
                print('nop')
        'Return the texts of listed versions as a list of strings.\n\n        Raises RevisionNotPresent if version is not present in\n        file history.\n        '
        return [''.join(self.get_lines(v)) for v in version_ids]

    def get_lines(self, version_id):
        if False:
            print('Hello World!')
        'Return version contents as a sequence of lines.\n\n        Raises RevisionNotPresent if version is not present in\n        file history.\n        '
        raise NotImplementedError(self.get_lines)

    def _get_lf_split_line_list(self, version_ids):
        if False:
            for i in range(10):
                print('nop')
        return [StringIO(t).readlines() for t in self.get_texts(version_ids)]

    def get_ancestry(self, version_ids, topo_sorted=True):
        if False:
            return 10
        'Return a list of all ancestors of given version(s). This\n        will not include the null revision.\n\n        This list will not be topologically sorted if topo_sorted=False is\n        passed.\n\n        Must raise RevisionNotPresent if any of the given versions are\n        not present in file history.'
        if isinstance(version_ids, basestring):
            version_ids = [version_ids]
        raise NotImplementedError(self.get_ancestry)

    def get_ancestry_with_ghosts(self, version_ids):
        if False:
            while True:
                i = 10
        'Return a list of all ancestors of given version(s). This\n        will not include the null revision.\n\n        Must raise RevisionNotPresent if any of the given versions are\n        not present in file history.\n\n        Ghosts that are known about will be included in ancestry list,\n        but are not explicitly marked.\n        '
        raise NotImplementedError(self.get_ancestry_with_ghosts)

    def get_parent_map(self, version_ids):
        if False:
            while True:
                i = 10
        'Get a map of the parents of version_ids.\n\n        :param version_ids: The version ids to look up parents for.\n        :return: A mapping from version id to parents.\n        '
        raise NotImplementedError(self.get_parent_map)

    def get_parents_with_ghosts(self, version_id):
        if False:
            i = 10
            return i + 15
        'Return version names for parents of version_id.\n\n        Will raise RevisionNotPresent if version_id is not present\n        in the history.\n\n        Ghosts that are known about will be included in the parent list,\n        but are not explicitly marked.\n        '
        try:
            return list(self.get_parent_map([version_id])[version_id])
        except KeyError:
            raise errors.RevisionNotPresent(version_id, self)

    def annotate(self, version_id):
        if False:
            while True:
                i = 10
        'Return a list of (version-id, line) tuples for version_id.\n\n        :raise RevisionNotPresent: If the given version is\n        not present in file history.\n        '
        raise NotImplementedError(self.annotate)

    def iter_lines_added_or_present_in_versions(self, version_ids=None, pb=None):
        if False:
            return 10
        'Iterate over the lines in the versioned file from version_ids.\n\n        This may return lines from other versions. Each item the returned\n        iterator yields is a tuple of a line and a text version that that line\n        is present in (not introduced in).\n\n        Ordering of results is in whatever order is most suitable for the\n        underlying storage format.\n\n        If a progress bar is supplied, it may be used to indicate progress.\n        The caller is responsible for cleaning up progress bars (because this\n        is an iterator).\n\n        NOTES: Lines are normalised: they will all have \n terminators.\n               Lines are returned in arbitrary order.\n\n        :return: An iterator over (line, version_id).\n        '
        raise NotImplementedError(self.iter_lines_added_or_present_in_versions)

    def plan_merge(self, ver_a, ver_b):
        if False:
            print('Hello World!')
        'Return pseudo-annotation indicating how the two versions merge.\n\n        This is computed between versions a and b and their common\n        base.\n\n        Weave lines present in none of them are skipped entirely.\n\n        Legend:\n        killed-base Dead in base revision\n        killed-both Killed in each revision\n        killed-a    Killed in a\n        killed-b    Killed in b\n        unchanged   Alive in both a and b (possibly created in both)\n        new-a       Created in a\n        new-b       Created in b\n        ghost-a     Killed in a, unborn in b\n        ghost-b     Killed in b, unborn in a\n        irrelevant  Not in either revision\n        '
        raise NotImplementedError(VersionedFile.plan_merge)

    def weave_merge(self, plan, a_marker=TextMerge.A_MARKER, b_marker=TextMerge.B_MARKER):
        if False:
            for i in range(10):
                print('nop')
        return PlanWeaveMerge(plan, a_marker, b_marker).merge_lines()[0]

class RecordingVersionedFilesDecorator(object):
    """A minimal versioned files that records calls made on it.

    Only enough methods have been added to support tests using it to date.

    :ivar calls: A list of the calls made; can be reset at any time by
        assigning [] to it.
    """

    def __init__(self, backing_vf):
        if False:
            i = 10
            return i + 15
        'Create a RecordingVersionedFilesDecorator decorating backing_vf.\n\n        :param backing_vf: The versioned file to answer all methods.\n        '
        self._backing_vf = backing_vf
        self.calls = []

    def add_lines(self, key, parents, lines, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False, check_content=True):
        if False:
            while True:
                i = 10
        self.calls.append(('add_lines', key, parents, lines, parent_texts, left_matching_blocks, nostore_sha, random_id, check_content))
        return self._backing_vf.add_lines(key, parents, lines, parent_texts, left_matching_blocks, nostore_sha, random_id, check_content)

    def check(self):
        if False:
            i = 10
            return i + 15
        self._backing_vf.check()

    def get_parent_map(self, keys):
        if False:
            i = 10
            return i + 15
        self.calls.append(('get_parent_map', copy(keys)))
        return self._backing_vf.get_parent_map(keys)

    def get_record_stream(self, keys, sort_order, include_delta_closure):
        if False:
            return 10
        self.calls.append(('get_record_stream', list(keys), sort_order, include_delta_closure))
        return self._backing_vf.get_record_stream(keys, sort_order, include_delta_closure)

    def get_sha1s(self, keys):
        if False:
            return 10
        self.calls.append(('get_sha1s', copy(keys)))
        return self._backing_vf.get_sha1s(keys)

    def iter_lines_added_or_present_in_keys(self, keys, pb=None):
        if False:
            for i in range(10):
                print('nop')
        self.calls.append(('iter_lines_added_or_present_in_keys', copy(keys)))
        return self._backing_vf.iter_lines_added_or_present_in_keys(keys, pb=pb)

    def keys(self):
        if False:
            return 10
        self.calls.append(('keys',))
        return self._backing_vf.keys()

class OrderingVersionedFilesDecorator(RecordingVersionedFilesDecorator):
    """A VF that records calls, and returns keys in specific order.

    :ivar calls: A list of the calls made; can be reset at any time by
        assigning [] to it.
    """

    def __init__(self, backing_vf, key_priority):
        if False:
            for i in range(10):
                print('nop')
        "Create a RecordingVersionedFilesDecorator decorating backing_vf.\n\n        :param backing_vf: The versioned file to answer all methods.\n        :param key_priority: A dictionary defining what order keys should be\n            returned from an 'unordered' get_record_stream request.\n            Keys with lower priority are returned first, keys not present in\n            the map get an implicit priority of 0, and are returned in\n            lexicographical order.\n        "
        RecordingVersionedFilesDecorator.__init__(self, backing_vf)
        self._key_priority = key_priority

    def get_record_stream(self, keys, sort_order, include_delta_closure):
        if False:
            return 10
        self.calls.append(('get_record_stream', list(keys), sort_order, include_delta_closure))
        if sort_order == 'unordered':

            def sort_key(key):
                if False:
                    for i in range(10):
                        print('nop')
                return (self._key_priority.get(key, 0), key)
            for key in sorted(keys, key=sort_key):
                for record in self._backing_vf.get_record_stream([key], 'unordered', include_delta_closure):
                    yield record
        else:
            for record in self._backing_vf.get_record_stream(keys, sort_order, include_delta_closure):
                yield record

class KeyMapper(object):
    """KeyMappers map between keys and underlying partitioned storage."""

    def map(self, key):
        if False:
            print('Hello World!')
        "Map key to an underlying storage identifier.\n\n        :param key: A key tuple e.g. ('file-id', 'revision-id').\n        :return: An underlying storage identifier, specific to the partitioning\n            mechanism.\n        "
        raise NotImplementedError(self.map)

    def unmap(self, partition_id):
        if False:
            for i in range(10):
                print('nop')
        'Map a partitioned storage id back to a key prefix.\n\n        :param partition_id: The underlying partition id.\n        :return: As much of a key (or prefix) as is derivable from the partition\n            id.\n        '
        raise NotImplementedError(self.unmap)

class ConstantMapper(KeyMapper):
    """A key mapper that maps to a constant result."""

    def __init__(self, result):
        if False:
            print('Hello World!')
        'Create a ConstantMapper which will return result for all maps.'
        self._result = result

    def map(self, key):
        if False:
            i = 10
            return i + 15
        'See KeyMapper.map().'
        return self._result

class URLEscapeMapper(KeyMapper):
    """Base class for use with transport backed storage.

    This provides a map and unmap wrapper that respectively url escape and
    unescape their outputs and inputs.
    """

    def map(self, key):
        if False:
            for i in range(10):
                print('nop')
        'See KeyMapper.map().'
        return urlutils.quote(self._map(key))

    def unmap(self, partition_id):
        if False:
            for i in range(10):
                print('nop')
        'See KeyMapper.unmap().'
        return self._unmap(urlutils.unquote(partition_id))

class PrefixMapper(URLEscapeMapper):
    """A key mapper that extracts the first component of a key.

    This mapper is for use with a transport based backend.
    """

    def _map(self, key):
        if False:
            return 10
        'See KeyMapper.map().'
        return key[0]

    def _unmap(self, partition_id):
        if False:
            print('Hello World!')
        'See KeyMapper.unmap().'
        return (partition_id,)

class HashPrefixMapper(URLEscapeMapper):
    """A key mapper that combines the first component of a key with a hash.

    This mapper is for use with a transport based backend.
    """

    def _map(self, key):
        if False:
            return 10
        'See KeyMapper.map().'
        prefix = self._escape(key[0])
        return '%02x/%s' % (adler32(prefix) & 255, prefix)

    def _escape(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        'No escaping needed here.'
        return prefix

    def _unmap(self, partition_id):
        if False:
            return 10
        'See KeyMapper.unmap().'
        return (self._unescape(osutils.basename(partition_id)),)

    def _unescape(self, basename):
        if False:
            while True:
                i = 10
        'No unescaping needed for HashPrefixMapper.'
        return basename

class HashEscapedPrefixMapper(HashPrefixMapper):
    """Combines the escaped first component of a key with a hash.

    This mapper is for use with a transport based backend.
    """
    _safe = 'abcdefghijklmnopqrstuvwxyz0123456789-_@,.'

    def _escape(self, prefix):
        if False:
            return 10
        "Turn a key element into a filesystem safe string.\n\n        This is similar to a plain urlutils.quote, except\n        it uses specific safe characters, so that it doesn't\n        have to translate a lot of valid file ids.\n        "
        r = [c in self._safe and c or '%%%02x' % ord(c) for c in prefix]
        return ''.join(r)

    def _unescape(self, basename):
        if False:
            return 10
        'Escaped names are easily unescaped by urlutils.'
        return urlutils.unquote(basename)

def make_versioned_files_factory(versioned_file_factory, mapper):
    if False:
        while True:
            i = 10
    'Create a ThunkedVersionedFiles factory.\n\n    This will create a callable which when called creates a\n    ThunkedVersionedFiles on a transport, using mapper to access individual\n    versioned files, and versioned_file_factory to create each individual file.\n    '

    def factory(transport):
        if False:
            i = 10
            return i + 15
        return ThunkedVersionedFiles(transport, versioned_file_factory, mapper, lambda : True)
    return factory

class VersionedFiles(object):
    """Storage for many versioned files.

    This object allows a single keyspace for accessing the history graph and
    contents of named bytestrings.

    Currently no implementation allows the graph of different key prefixes to
    intersect, but the API does allow such implementations in the future.

    The keyspace is expressed via simple tuples. Any instance of VersionedFiles
    may have a different length key-size, but that size will be constant for
    all texts added to or retrieved from it. For instance, bzrlib uses
    instances with a key-size of 2 for storing user files in a repository, with
    the first element the fileid, and the second the version of that file.

    The use of tuples allows a single code base to support several different
    uses with only the mapping logic changing from instance to instance.

    :ivar _immediate_fallback_vfs: For subclasses that support stacking,
        this is a list of other VersionedFiles immediately underneath this
        one.  They may in turn each have further fallbacks.
    """

    def add_lines(self, key, parents, lines, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False, check_content=True):
        if False:
            for i in range(10):
                print('nop')
        "Add a text to the store.\n\n        :param key: The key tuple of the text to add. If the last element is\n            None, a CHK string will be generated during the addition.\n        :param parents: The parents key tuples of the text to add.\n        :param lines: A list of lines. Each line must be a bytestring. And all\n            of them except the last must be terminated with \n and contain no\n            other \n's. The last line may either contain no \n's or a single\n            terminating \n. If the lines list does meet this constraint the add\n            routine may error or may succeed - but you will be unable to read\n            the data back accurately. (Checking the lines have been split\n            correctly is expensive and extremely unlikely to catch bugs so it\n            is not done at runtime unless check_content is True.)\n        :param parent_texts: An optional dictionary containing the opaque\n            representations of some or all of the parents of version_id to\n            allow delta optimisations.  VERY IMPORTANT: the texts must be those\n            returned by add_lines or data corruption can be caused.\n        :param left_matching_blocks: a hint about which areas are common\n            between the text and its left-hand-parent.  The format is\n            the SequenceMatcher.get_matching_blocks format.\n        :param nostore_sha: Raise ExistingContent and do not add the lines to\n            the versioned file if the digest of the lines matches this.\n        :param random_id: If True a random id has been selected rather than\n            an id determined by some deterministic process such as a converter\n            from a foreign VCS. When True the backend may choose not to check\n            for uniqueness of the resulting key within the versioned file, so\n            this should only be done when the result is expected to be unique\n            anyway.\n        :param check_content: If True, the lines supplied are verified to be\n            bytestrings that are correctly formed lines.\n        :return: The text sha1, the number of bytes in the text, and an opaque\n                 representation of the inserted version which can be provided\n                 back to future add_lines calls in the parent_texts dictionary.\n        "
        raise NotImplementedError(self.add_lines)

    def _add_text(self, key, parents, text, nostore_sha=None, random_id=False):
        if False:
            print('Hello World!')
        'Add a text to the store.\n\n        This is a private function for use by VersionedFileCommitBuilder.\n\n        :param key: The key tuple of the text to add. If the last element is\n            None, a CHK string will be generated during the addition.\n        :param parents: The parents key tuples of the text to add.\n        :param text: A string containing the text to be committed.\n        :param nostore_sha: Raise ExistingContent and do not add the lines to\n            the versioned file if the digest of the lines matches this.\n        :param random_id: If True a random id has been selected rather than\n            an id determined by some deterministic process such as a converter\n            from a foreign VCS. When True the backend may choose not to check\n            for uniqueness of the resulting key within the versioned file, so\n            this should only be done when the result is expected to be unique\n            anyway.\n        :param check_content: If True, the lines supplied are verified to be\n            bytestrings that are correctly formed lines.\n        :return: The text sha1, the number of bytes in the text, and an opaque\n                 representation of the inserted version which can be provided\n                 back to future _add_text calls in the parent_texts dictionary.\n        '
        return self.add_lines(key, parents, osutils.split_lines(text), nostore_sha=nostore_sha, random_id=random_id, check_content=True)

    def add_mpdiffs(self, records):
        if False:
            i = 10
            return i + 15
        'Add mpdiffs to this VersionedFile.\n\n        Records should be iterables of version, parents, expected_sha1,\n        mpdiff. mpdiff should be a MultiParent instance.\n        '
        vf_parents = {}
        mpvf = multiparent.MultiMemoryVersionedFile()
        versions = []
        for (version, parent_ids, expected_sha1, mpdiff) in records:
            versions.append(version)
            mpvf.add_diff(mpdiff, version, parent_ids)
        needed_parents = set()
        for (version, parent_ids, expected_sha1, mpdiff) in records:
            needed_parents.update((p for p in parent_ids if not mpvf.has_version(p)))
        chunks_to_lines = osutils.chunks_to_lines
        for record in self.get_record_stream(needed_parents, 'unordered', True):
            if record.storage_kind == 'absent':
                continue
            mpvf.add_version(chunks_to_lines(record.get_bytes_as('chunked')), record.key, [])
        for ((key, parent_keys, expected_sha1, mpdiff), lines) in zip(records, mpvf.get_line_list(versions)):
            if len(parent_keys) == 1:
                left_matching_blocks = list(mpdiff.get_matching_blocks(0, mpvf.get_diff(parent_keys[0]).num_lines()))
            else:
                left_matching_blocks = None
            (version_sha1, _, version_text) = self.add_lines(key, parent_keys, lines, vf_parents, left_matching_blocks=left_matching_blocks)
            if version_sha1 != expected_sha1:
                raise errors.VersionedFileInvalidChecksum(version)
            vf_parents[key] = version_text

    def annotate(self, key):
        if False:
            i = 10
            return i + 15
        'Return a list of (version-key, line) tuples for the text of key.\n\n        :raise RevisionNotPresent: If the key is not present.\n        '
        raise NotImplementedError(self.annotate)

    def check(self, progress_bar=None):
        if False:
            i = 10
            return i + 15
        'Check this object for integrity.\n        \n        :param progress_bar: A progress bar to output as the check progresses.\n        :param keys: Specific keys within the VersionedFiles to check. When\n            this parameter is not None, check() becomes a generator as per\n            get_record_stream. The difference to get_record_stream is that\n            more or deeper checks will be performed.\n        :return: None, or if keys was supplied a generator as per\n            get_record_stream.\n        '
        raise NotImplementedError(self.check)

    @staticmethod
    def check_not_reserved_id(version_id):
        if False:
            return 10
        revision.check_not_reserved_id(version_id)

    def clear_cache(self):
        if False:
            i = 10
            return i + 15
        "Clear whatever caches this VersionedFile holds.\n\n        This is generally called after an operation has been performed, when we\n        don't expect to be using this versioned file again soon.\n        "

    def _check_lines_not_unicode(self, lines):
        if False:
            while True:
                i = 10
        'Check that lines being added to a versioned file are not unicode.'
        for line in lines:
            if line.__class__ is not str:
                raise errors.BzrBadParameterUnicode('lines')

    def _check_lines_are_lines(self, lines):
        if False:
            print('Hello World!')
        'Check that the lines really are full lines without inline EOL.'
        for line in lines:
            if '\n' in line[:-1]:
                raise errors.BzrBadParameterContainsNewline('lines')

    def get_known_graph_ancestry(self, keys):
        if False:
            i = 10
            return i + 15
        'Get a KnownGraph instance with the ancestry of keys.'
        pending = set(keys)
        parent_map = {}
        while pending:
            this_parent_map = self.get_parent_map(pending)
            parent_map.update(this_parent_map)
            pending = set()
            map(pending.update, this_parent_map.itervalues())
            pending = pending.difference(parent_map)
        kg = _mod_graph.KnownGraph(parent_map)
        return kg

    def get_parent_map(self, keys):
        if False:
            while True:
                i = 10
        'Get a map of the parents of keys.\n\n        :param keys: The keys to look up parents for.\n        :return: A mapping from keys to parents. Absent keys are absent from\n            the mapping.\n        '
        raise NotImplementedError(self.get_parent_map)

    def get_record_stream(self, keys, ordering, include_delta_closure):
        if False:
            return 10
        "Get a stream of records for keys.\n\n        :param keys: The keys to include.\n        :param ordering: Either 'unordered' or 'topological'. A topologically\n            sorted stream has compression parents strictly before their\n            children.\n        :param include_delta_closure: If True then the closure across any\n            compression parents will be included (in the opaque data).\n        :return: An iterator of ContentFactory objects, each of which is only\n            valid until the iterator is advanced.\n        "
        raise NotImplementedError(self.get_record_stream)

    def get_sha1s(self, keys):
        if False:
            print('Hello World!')
        "Get the sha1's of the texts for the given keys.\n\n        :param keys: The names of the keys to lookup\n        :return: a dict from key to sha1 digest. Keys of texts which are not\n            present in the store are not present in the returned\n            dictionary.\n        "
        raise NotImplementedError(self.get_sha1s)
    has_key = index._has_key_from_parent_map

    def get_missing_compression_parent_keys(self):
        if False:
            print('Hello World!')
        'Return an iterable of keys of missing compression parents.\n\n        Check this after calling insert_record_stream to find out if there are\n        any missing compression parents.  If there are, the records that\n        depend on them are not able to be inserted safely. The precise\n        behaviour depends on the concrete VersionedFiles class in use.\n\n        Classes that do not support this will raise NotImplementedError.\n        '
        raise NotImplementedError(self.get_missing_compression_parent_keys)

    def insert_record_stream(self, stream):
        if False:
            print('Hello World!')
        'Insert a record stream into this container.\n\n        :param stream: A stream of records to insert.\n        :return: None\n        :seealso VersionedFile.get_record_stream:\n        '
        raise NotImplementedError

    def iter_lines_added_or_present_in_keys(self, keys, pb=None):
        if False:
            i = 10
            return i + 15
        'Iterate over the lines in the versioned files from keys.\n\n        This may return lines from other keys. Each item the returned\n        iterator yields is a tuple of a line and a text version that that line\n        is present in (not introduced in).\n\n        Ordering of results is in whatever order is most suitable for the\n        underlying storage format.\n\n        If a progress bar is supplied, it may be used to indicate progress.\n        The caller is responsible for cleaning up progress bars (because this\n        is an iterator).\n\n        NOTES:\n         * Lines are normalised by the underlying store: they will all have \n\n           terminators.\n         * Lines are returned in arbitrary order.\n\n        :return: An iterator over (line, key).\n        '
        raise NotImplementedError(self.iter_lines_added_or_present_in_keys)

    def keys(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a iterable of the keys for all the contained texts.'
        raise NotImplementedError(self.keys)

    def make_mpdiffs(self, keys):
        if False:
            while True:
                i = 10
        'Create multiparent diffs for specified keys.'
        generator = _MPDiffGenerator(self, keys)
        return generator.compute_diffs()

    def get_annotator(self):
        if False:
            for i in range(10):
                print('nop')
        return annotate.Annotator(self)
    missing_keys = index._missing_keys_from_parent_map

    def _extract_blocks(self, version_id, source, target):
        if False:
            print('Hello World!')
        return None

    def _transitive_fallbacks(self):
        if False:
            for i in range(10):
                print('nop')
        "Return the whole stack of fallback versionedfiles.\n\n        This VersionedFiles may have a list of fallbacks, but it doesn't\n        necessarily know about the whole stack going down, and it can't know\n        at open time because they may change after the objects are opened.\n        "
        all_fallbacks = []
        for a_vfs in self._immediate_fallback_vfs:
            all_fallbacks.append(a_vfs)
            all_fallbacks.extend(a_vfs._transitive_fallbacks())
        return all_fallbacks

class ThunkedVersionedFiles(VersionedFiles):
    """Storage for many versioned files thunked onto a 'VersionedFile' class.

    This object allows a single keyspace for accessing the history graph and
    contents of named bytestrings.

    Currently no implementation allows the graph of different key prefixes to
    intersect, but the API does allow such implementations in the future.
    """

    def __init__(self, transport, file_factory, mapper, is_locked):
        if False:
            print('Hello World!')
        'Create a ThunkedVersionedFiles.'
        self._transport = transport
        self._file_factory = file_factory
        self._mapper = mapper
        self._is_locked = is_locked

    def add_lines(self, key, parents, lines, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False, check_content=True):
        if False:
            for i in range(10):
                print('nop')
        'See VersionedFiles.add_lines().'
        path = self._mapper.map(key)
        version_id = key[-1]
        parents = [parent[-1] for parent in parents]
        vf = self._get_vf(path)
        try:
            try:
                return vf.add_lines_with_ghosts(version_id, parents, lines, parent_texts=parent_texts, left_matching_blocks=left_matching_blocks, nostore_sha=nostore_sha, random_id=random_id, check_content=check_content)
            except NotImplementedError:
                return vf.add_lines(version_id, parents, lines, parent_texts=parent_texts, left_matching_blocks=left_matching_blocks, nostore_sha=nostore_sha, random_id=random_id, check_content=check_content)
        except errors.NoSuchFile:
            self._transport.mkdir(osutils.dirname(path))
            try:
                return vf.add_lines_with_ghosts(version_id, parents, lines, parent_texts=parent_texts, left_matching_blocks=left_matching_blocks, nostore_sha=nostore_sha, random_id=random_id, check_content=check_content)
            except NotImplementedError:
                return vf.add_lines(version_id, parents, lines, parent_texts=parent_texts, left_matching_blocks=left_matching_blocks, nostore_sha=nostore_sha, random_id=random_id, check_content=check_content)

    def annotate(self, key):
        if False:
            i = 10
            return i + 15
        'Return a list of (version-key, line) tuples for the text of key.\n\n        :raise RevisionNotPresent: If the key is not present.\n        '
        prefix = key[:-1]
        path = self._mapper.map(prefix)
        vf = self._get_vf(path)
        origins = vf.annotate(key[-1])
        result = []
        for (origin, line) in origins:
            result.append((prefix + (origin,), line))
        return result

    def check(self, progress_bar=None, keys=None):
        if False:
            i = 10
            return i + 15
        'See VersionedFiles.check().'
        for (prefix, vf) in self._iter_all_components():
            vf.check()
        if keys is not None:
            return self.get_record_stream(keys, 'unordered', True)

    def get_parent_map(self, keys):
        if False:
            for i in range(10):
                print('nop')
        'Get a map of the parents of keys.\n\n        :param keys: The keys to look up parents for.\n        :return: A mapping from keys to parents. Absent keys are absent from\n            the mapping.\n        '
        prefixes = self._partition_keys(keys)
        result = {}
        for (prefix, suffixes) in prefixes.items():
            path = self._mapper.map(prefix)
            vf = self._get_vf(path)
            parent_map = vf.get_parent_map(suffixes)
            for (key, parents) in parent_map.items():
                result[prefix + (key,)] = tuple((prefix + (parent,) for parent in parents))
        return result

    def _get_vf(self, path):
        if False:
            return 10
        if not self._is_locked():
            raise errors.ObjectNotLocked(self)
        return self._file_factory(path, self._transport, create=True, get_scope=lambda : None)

    def _partition_keys(self, keys):
        if False:
            return 10
        'Turn keys into a dict of prefix:suffix_list.'
        result = {}
        for key in keys:
            prefix_keys = result.setdefault(key[:-1], [])
            prefix_keys.append(key[-1])
        return result

    def _get_all_prefixes(self):
        if False:
            i = 10
            return i + 15
        if type(self._mapper) == ConstantMapper:
            paths = [self._mapper.map(())]
            prefixes = [()]
        else:
            relpaths = set()
            for quoted_relpath in self._transport.iter_files_recursive():
                (path, ext) = os.path.splitext(quoted_relpath)
                relpaths.add(path)
            paths = list(relpaths)
            prefixes = [self._mapper.unmap(path) for path in paths]
        return zip(paths, prefixes)

    def get_record_stream(self, keys, ordering, include_delta_closure):
        if False:
            while True:
                i = 10
        'See VersionedFiles.get_record_stream().'
        keys = sorted(keys)
        for (prefix, suffixes, vf) in self._iter_keys_vf(keys):
            suffixes = [(suffix,) for suffix in suffixes]
            for record in vf.get_record_stream(suffixes, ordering, include_delta_closure):
                if record.parents is not None:
                    record.parents = tuple((prefix + parent for parent in record.parents))
                record.key = prefix + record.key
                yield record

    def _iter_keys_vf(self, keys):
        if False:
            print('Hello World!')
        prefixes = self._partition_keys(keys)
        sha1s = {}
        for (prefix, suffixes) in prefixes.items():
            path = self._mapper.map(prefix)
            vf = self._get_vf(path)
            yield (prefix, suffixes, vf)

    def get_sha1s(self, keys):
        if False:
            while True:
                i = 10
        'See VersionedFiles.get_sha1s().'
        sha1s = {}
        for (prefix, suffixes, vf) in self._iter_keys_vf(keys):
            vf_sha1s = vf.get_sha1s(suffixes)
            for (suffix, sha1) in vf_sha1s.iteritems():
                sha1s[prefix + (suffix,)] = sha1
        return sha1s

    def insert_record_stream(self, stream):
        if False:
            return 10
        'Insert a record stream into this container.\n\n        :param stream: A stream of records to insert.\n        :return: None\n        :seealso VersionedFile.get_record_stream:\n        '
        for record in stream:
            prefix = record.key[:-1]
            key = record.key[-1:]
            if record.parents is not None:
                parents = [parent[-1:] for parent in record.parents]
            else:
                parents = None
            thunk_record = AdapterFactory(key, parents, record)
            path = self._mapper.map(prefix)
            vf = self._get_vf(path)
            vf.insert_record_stream([thunk_record])

    def iter_lines_added_or_present_in_keys(self, keys, pb=None):
        if False:
            while True:
                i = 10
        'Iterate over the lines in the versioned files from keys.\n\n        This may return lines from other keys. Each item the returned\n        iterator yields is a tuple of a line and a text version that that line\n        is present in (not introduced in).\n\n        Ordering of results is in whatever order is most suitable for the\n        underlying storage format.\n\n        If a progress bar is supplied, it may be used to indicate progress.\n        The caller is responsible for cleaning up progress bars (because this\n        is an iterator).\n\n        NOTES:\n         * Lines are normalised by the underlying store: they will all have \n\n           terminators.\n         * Lines are returned in arbitrary order.\n\n        :return: An iterator over (line, key).\n        '
        for (prefix, suffixes, vf) in self._iter_keys_vf(keys):
            for (line, version) in vf.iter_lines_added_or_present_in_versions(suffixes):
                yield (line, prefix + (version,))

    def _iter_all_components(self):
        if False:
            i = 10
            return i + 15
        for (path, prefix) in self._get_all_prefixes():
            yield (prefix, self._get_vf(path))

    def keys(self):
        if False:
            while True:
                i = 10
        'See VersionedFiles.keys().'
        result = set()
        for (prefix, vf) in self._iter_all_components():
            for suffix in vf.versions():
                result.add(prefix + (suffix,))
        return result

class VersionedFilesWithFallbacks(VersionedFiles):

    def without_fallbacks(self):
        if False:
            return 10
        'Return a clone of this object without any fallbacks configured.'
        raise NotImplementedError(self.without_fallbacks)

    def add_fallback_versioned_files(self, a_versioned_files):
        if False:
            return 10
        'Add a source of texts for texts not present in this knit.\n\n        :param a_versioned_files: A VersionedFiles object.\n        '
        raise NotImplementedError(self.add_fallback_versioned_files)

    def get_known_graph_ancestry(self, keys):
        if False:
            return 10
        'Get a KnownGraph instance with the ancestry of keys.'
        (parent_map, missing_keys) = self._index.find_ancestry(keys)
        for fallback in self._transitive_fallbacks():
            if not missing_keys:
                break
            (f_parent_map, f_missing_keys) = fallback._index.find_ancestry(missing_keys)
            parent_map.update(f_parent_map)
            missing_keys = f_missing_keys
        kg = _mod_graph.KnownGraph(parent_map)
        return kg

class _PlanMergeVersionedFile(VersionedFiles):
    """A VersionedFile for uncommitted and committed texts.

    It is intended to allow merges to be planned with working tree texts.
    It implements only the small part of the VersionedFiles interface used by
    PlanMerge.  It falls back to multiple versionedfiles for data not stored in
    _PlanMergeVersionedFile itself.

    :ivar: fallback_versionedfiles a list of VersionedFiles objects that can be
        queried for missing texts.
    """

    def __init__(self, file_id):
        if False:
            return 10
        'Create a _PlanMergeVersionedFile.\n\n        :param file_id: Used with _PlanMerge code which is not yet fully\n            tuple-keyspace aware.\n        '
        self._file_id = file_id
        self.fallback_versionedfiles = []
        self._parents = {}
        self._lines = {}
        self._providers = [_mod_graph.DictParentsProvider(self._parents)]

    def plan_merge(self, ver_a, ver_b, base=None):
        if False:
            print('Hello World!')
        'See VersionedFile.plan_merge'
        from bzrlib.merge import _PlanMerge
        if base is None:
            return _PlanMerge(ver_a, ver_b, self, (self._file_id,)).plan_merge()
        old_plan = list(_PlanMerge(ver_a, base, self, (self._file_id,)).plan_merge())
        new_plan = list(_PlanMerge(ver_a, ver_b, self, (self._file_id,)).plan_merge())
        return _PlanMerge._subtract_plans(old_plan, new_plan)

    def plan_lca_merge(self, ver_a, ver_b, base=None):
        if False:
            print('Hello World!')
        from bzrlib.merge import _PlanLCAMerge
        graph = _mod_graph.Graph(self)
        new_plan = _PlanLCAMerge(ver_a, ver_b, self, (self._file_id,), graph).plan_merge()
        if base is None:
            return new_plan
        old_plan = _PlanLCAMerge(ver_a, base, self, (self._file_id,), graph).plan_merge()
        return _PlanLCAMerge._subtract_plans(list(old_plan), list(new_plan))

    def add_lines(self, key, parents, lines):
        if False:
            i = 10
            return i + 15
        'See VersionedFiles.add_lines\n\n        Lines are added locally, not to fallback versionedfiles.  Also, ghosts\n        are permitted.  Only reserved ids are permitted.\n        '
        if type(key) is not tuple:
            raise TypeError(key)
        if not revision.is_reserved_id(key[-1]):
            raise ValueError('Only reserved ids may be used')
        if parents is None:
            raise ValueError('Parents may not be None')
        if lines is None:
            raise ValueError('Lines may not be None')
        self._parents[key] = tuple(parents)
        self._lines[key] = lines

    def get_record_stream(self, keys, ordering, include_delta_closure):
        if False:
            i = 10
            return i + 15
        pending = set(keys)
        for key in keys:
            if key in self._lines:
                lines = self._lines[key]
                parents = self._parents[key]
                pending.remove(key)
                yield ChunkedContentFactory(key, parents, None, lines)
        for versionedfile in self.fallback_versionedfiles:
            for record in versionedfile.get_record_stream(pending, 'unordered', True):
                if record.storage_kind == 'absent':
                    continue
                else:
                    pending.remove(record.key)
                    yield record
            if not pending:
                return
        for key in pending:
            yield AbsentContentFactory(key)

    def get_parent_map(self, keys):
        if False:
            i = 10
            return i + 15
        'See VersionedFiles.get_parent_map'
        keys = set(keys)
        result = {}
        if revision.NULL_REVISION in keys:
            keys.remove(revision.NULL_REVISION)
            result[revision.NULL_REVISION] = ()
        self._providers = self._providers[:1] + self.fallback_versionedfiles
        result.update(_mod_graph.StackedParentsProvider(self._providers).get_parent_map(keys))
        for (key, parents) in result.iteritems():
            if parents == ():
                result[key] = (revision.NULL_REVISION,)
        return result

class PlanWeaveMerge(TextMerge):
    """Weave merge that takes a plan as its input.

    This exists so that VersionedFile.plan_merge is implementable.
    Most callers will want to use WeaveMerge instead.
    """

    def __init__(self, plan, a_marker=TextMerge.A_MARKER, b_marker=TextMerge.B_MARKER):
        if False:
            for i in range(10):
                print('nop')
        TextMerge.__init__(self, a_marker, b_marker)
        self.plan = list(plan)

    def _merge_struct(self):
        if False:
            for i in range(10):
                print('nop')
        lines_a = []
        lines_b = []
        ch_a = ch_b = False

        def outstanding_struct():
            if False:
                print('Hello World!')
            if not lines_a and (not lines_b):
                return
            elif ch_a and (not ch_b):
                yield (lines_a,)
            elif ch_b and (not ch_a):
                yield (lines_b,)
            elif lines_a == lines_b:
                yield (lines_a,)
            else:
                yield (lines_a, lines_b)
        for (state, line) in self.plan:
            if state == 'unchanged':
                for struct in outstanding_struct():
                    yield struct
                lines_a = []
                lines_b = []
                ch_a = ch_b = False
            if state == 'unchanged':
                if line:
                    yield ([line],)
            elif state == 'killed-a':
                ch_a = True
                lines_b.append(line)
            elif state == 'killed-b':
                ch_b = True
                lines_a.append(line)
            elif state == 'new-a':
                ch_a = True
                lines_a.append(line)
            elif state == 'new-b':
                ch_b = True
                lines_b.append(line)
            elif state == 'conflicted-a':
                ch_b = ch_a = True
                lines_a.append(line)
            elif state == 'conflicted-b':
                ch_b = ch_a = True
                lines_b.append(line)
            elif state == 'killed-both':
                ch_b = ch_a = True
            elif state not in ('irrelevant', 'ghost-a', 'ghost-b', 'killed-base'):
                raise AssertionError(state)
        for struct in outstanding_struct():
            yield struct

    def base_from_plan(self):
        if False:
            for i in range(10):
                print('nop')
        'Construct a BASE file from the plan text.'
        base_lines = []
        for (state, line) in self.plan:
            if state in ('killed-a', 'killed-b', 'killed-both', 'unchanged'):
                base_lines.append(line)
            elif state not in ('killed-base', 'irrelevant', 'ghost-a', 'ghost-b', 'new-a', 'new-b', 'conflicted-a', 'conflicted-b'):
                raise AssertionError('Unknown state: %s' % (state,))
        return base_lines

class WeaveMerge(PlanWeaveMerge):
    """Weave merge that takes a VersionedFile and two versions as its input."""

    def __init__(self, versionedfile, ver_a, ver_b, a_marker=PlanWeaveMerge.A_MARKER, b_marker=PlanWeaveMerge.B_MARKER):
        if False:
            i = 10
            return i + 15
        plan = versionedfile.plan_merge(ver_a, ver_b)
        PlanWeaveMerge.__init__(self, plan, a_marker, b_marker)

class VirtualVersionedFiles(VersionedFiles):
    """Dummy implementation for VersionedFiles that uses other functions for
    obtaining fulltexts and parent maps.

    This is always on the bottom of the stack and uses string keys
    (rather than tuples) internally.
    """

    def __init__(self, get_parent_map, get_lines):
        if False:
            i = 10
            return i + 15
        'Create a VirtualVersionedFiles.\n\n        :param get_parent_map: Same signature as Repository.get_parent_map.\n        :param get_lines: Should return lines for specified key or None if\n                          not available.\n        '
        super(VirtualVersionedFiles, self).__init__()
        self._get_parent_map = get_parent_map
        self._get_lines = get_lines

    def check(self, progressbar=None):
        if False:
            while True:
                i = 10
        'See VersionedFiles.check.\n\n        :note: Always returns True for VirtualVersionedFiles.\n        '
        return True

    def add_mpdiffs(self, records):
        if False:
            print('Hello World!')
        'See VersionedFiles.mpdiffs.\n\n        :note: Not implemented for VirtualVersionedFiles.\n        '
        raise NotImplementedError(self.add_mpdiffs)

    def get_parent_map(self, keys):
        if False:
            while True:
                i = 10
        'See VersionedFiles.get_parent_map.'
        return dict([((k,), tuple([(p,) for p in v])) for (k, v) in self._get_parent_map([k for (k,) in keys]).iteritems()])

    def get_sha1s(self, keys):
        if False:
            while True:
                i = 10
        'See VersionedFiles.get_sha1s.'
        ret = {}
        for (k,) in keys:
            lines = self._get_lines(k)
            if lines is not None:
                if not isinstance(lines, list):
                    raise AssertionError
                ret[k,] = osutils.sha_strings(lines)
        return ret

    def get_record_stream(self, keys, ordering, include_delta_closure):
        if False:
            i = 10
            return i + 15
        'See VersionedFiles.get_record_stream.'
        for (k,) in list(keys):
            lines = self._get_lines(k)
            if lines is not None:
                if not isinstance(lines, list):
                    raise AssertionError
                yield ChunkedContentFactory((k,), None, sha1=osutils.sha_strings(lines), chunks=lines)
            else:
                yield AbsentContentFactory((k,))

    def iter_lines_added_or_present_in_keys(self, keys, pb=None):
        if False:
            i = 10
            return i + 15
        'See VersionedFile.iter_lines_added_or_present_in_versions().'
        for (i, (key,)) in enumerate(keys):
            if pb is not None:
                pb.update('Finding changed lines', i, len(keys))
            for l in self._get_lines(key):
                yield (l, key)

class NoDupeAddLinesDecorator(object):
    """Decorator for a VersionedFiles that skips doing an add_lines if the key
    is already present.
    """

    def __init__(self, store):
        if False:
            print('Hello World!')
        self._store = store

    def add_lines(self, key, parents, lines, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False, check_content=True):
        if False:
            i = 10
            return i + 15
        "See VersionedFiles.add_lines.\n        \n        This implementation may return None as the third element of the return\n        value when the original store wouldn't.\n        "
        if nostore_sha:
            raise NotImplementedError('NoDupeAddLinesDecorator.add_lines does not implement the nostore_sha behaviour.')
        if key[-1] is None:
            sha1 = osutils.sha_strings(lines)
            key = ('sha1:' + sha1,)
        else:
            sha1 = None
        if key in self._store.get_parent_map([key]):
            if sha1 is None:
                sha1 = osutils.sha_strings(lines)
            return (sha1, sum(map(len, lines)), None)
        return self._store.add_lines(key, parents, lines, parent_texts=parent_texts, left_matching_blocks=left_matching_blocks, nostore_sha=nostore_sha, random_id=random_id, check_content=check_content)

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        return getattr(self._store, name)

def network_bytes_to_kind_and_offset(network_bytes):
    if False:
        i = 10
        return i + 15
    'Strip of a record kind from the front of network_bytes.\n\n    :param network_bytes: The bytes of a record.\n    :return: A tuple (storage_kind, offset_of_remaining_bytes)\n    '
    line_end = network_bytes.find('\n')
    storage_kind = network_bytes[:line_end]
    return (storage_kind, line_end + 1)

class NetworkRecordStream(object):
    """A record_stream which reconstitures a serialised stream."""

    def __init__(self, bytes_iterator):
        if False:
            i = 10
            return i + 15
        "Create a NetworkRecordStream.\n\n        :param bytes_iterator: An iterator of bytes. Each item in this\n            iterator should have been obtained from a record_streams'\n            record.get_bytes_as(record.storage_kind) call.\n        "
        self._bytes_iterator = bytes_iterator
        self._kind_factory = {'fulltext': fulltext_network_to_record, 'groupcompress-block': groupcompress.network_block_to_records, 'knit-ft-gz': knit.knit_network_to_record, 'knit-delta-gz': knit.knit_network_to_record, 'knit-annotated-ft-gz': knit.knit_network_to_record, 'knit-annotated-delta-gz': knit.knit_network_to_record, 'knit-delta-closure': knit.knit_delta_closure_to_records}

    def read(self):
        if False:
            for i in range(10):
                print('nop')
        'Read the stream.\n\n        :return: An iterator as per VersionedFiles.get_record_stream().\n        '
        for bytes in self._bytes_iterator:
            (storage_kind, line_end) = network_bytes_to_kind_and_offset(bytes)
            for record in self._kind_factory[storage_kind](storage_kind, bytes, line_end):
                yield record

def fulltext_network_to_record(kind, bytes, line_end):
    if False:
        return 10
    'Convert a network fulltext record to record.'
    (meta_len,) = struct.unpack('!L', bytes[line_end:line_end + 4])
    record_meta = bytes[line_end + 4:line_end + 4 + meta_len]
    (key, parents) = bencode.bdecode_as_tuple(record_meta)
    if parents == 'nil':
        parents = None
    fulltext = bytes[line_end + 4 + meta_len:]
    return [FulltextContentFactory(key, parents, None, fulltext)]

def _length_prefix(bytes):
    if False:
        while True:
            i = 10
    return struct.pack('!L', len(bytes))

def record_to_fulltext_bytes(record):
    if False:
        i = 10
        return i + 15
    if record.parents is None:
        parents = 'nil'
    else:
        parents = record.parents
    record_meta = bencode.bencode((record.key, parents))
    record_content = record.get_bytes_as('fulltext')
    return 'fulltext\n%s%s%s' % (_length_prefix(record_meta), record_meta, record_content)

def sort_groupcompress(parent_map):
    if False:
        for i in range(10):
            print('nop')
    'Sort and group the keys in parent_map into groupcompress order.\n\n    groupcompress is defined (currently) as reverse-topological order, grouped\n    by the key prefix.\n\n    :return: A sorted-list of keys\n    '
    per_prefix_map = {}
    for item in parent_map.iteritems():
        key = item[0]
        if isinstance(key, str) or len(key) == 1:
            prefix = ''
        else:
            prefix = key[0]
        try:
            per_prefix_map[prefix].append(item)
        except KeyError:
            per_prefix_map[prefix] = [item]
    present_keys = []
    for prefix in sorted(per_prefix_map):
        present_keys.extend(reversed(tsort.topo_sort(per_prefix_map[prefix])))
    return present_keys

class _KeyRefs(object):

    def __init__(self, track_new_keys=False):
        if False:
            while True:
                i = 10
        self.refs = {}
        if track_new_keys:
            self.new_keys = set()
        else:
            self.new_keys = None

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        if self.refs:
            self.refs.clear()
        if self.new_keys:
            self.new_keys.clear()

    def add_references(self, key, refs):
        if False:
            while True:
                i = 10
        for referenced in refs:
            try:
                needed_by = self.refs[referenced]
            except KeyError:
                needed_by = self.refs[referenced] = set()
            needed_by.add(key)
        self.add_key(key)

    def get_new_keys(self):
        if False:
            for i in range(10):
                print('nop')
        return self.new_keys

    def get_unsatisfied_refs(self):
        if False:
            i = 10
            return i + 15
        return self.refs.iterkeys()

    def _satisfy_refs_for_key(self, key):
        if False:
            while True:
                i = 10
        try:
            del self.refs[key]
        except KeyError:
            pass

    def add_key(self, key):
        if False:
            for i in range(10):
                print('nop')
        self._satisfy_refs_for_key(key)
        if self.new_keys is not None:
            self.new_keys.add(key)

    def satisfy_refs_for_keys(self, keys):
        if False:
            while True:
                i = 10
        for key in keys:
            self._satisfy_refs_for_key(key)

    def get_referrers(self):
        if False:
            return 10
        result = set()
        for referrers in self.refs.itervalues():
            result.update(referrers)
        return result