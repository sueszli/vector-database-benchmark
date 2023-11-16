"""Searching in versioned file repositories."""
from __future__ import absolute_import
from bzrlib import debug, revision, trace
from bzrlib.graph import DictParentsProvider, Graph, invert_parent_map

class AbstractSearchResult(object):
    """The result of a search, describing a set of keys.
    
    Search results are typically used as the 'fetch_spec' parameter when
    fetching revisions.

    :seealso: AbstractSearch
    """

    def get_recipe(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a recipe that can be used to replay this search.\n\n        The recipe allows reconstruction of the same results at a later date.\n\n        :return: A tuple of `(search_kind_str, *details)`.  The details vary by\n            kind of search result.\n        '
        raise NotImplementedError(self.get_recipe)

    def get_network_struct(self):
        if False:
            while True:
                i = 10
        'Return a tuple that can be transmitted via the HPSS protocol.'
        raise NotImplementedError(self.get_network_struct)

    def get_keys(self):
        if False:
            return 10
        'Return the keys found in this search.\n\n        :return: A set of keys.\n        '
        raise NotImplementedError(self.get_keys)

    def is_empty(self):
        if False:
            for i in range(10):
                print('nop')
        'Return false if the search lists 1 or more revisions.'
        raise NotImplementedError(self.is_empty)

    def refine(self, seen, referenced):
        if False:
            for i in range(10):
                print('nop')
        'Create a new search by refining this search.\n\n        :param seen: Revisions that have been satisfied.\n        :param referenced: Revision references observed while satisfying some\n            of this search.\n        :return: A search result.\n        '
        raise NotImplementedError(self.refine)

class AbstractSearch(object):
    """A search that can be executed, producing a search result.

    :seealso: AbstractSearchResult
    """

    def execute(self):
        if False:
            return 10
        "Construct a network-ready search result from this search description.\n\n        This may take some time to search repositories, etc.\n\n        :return: A search result (an object that implements\n            AbstractSearchResult's API).\n        "
        raise NotImplementedError(self.execute)

class SearchResult(AbstractSearchResult):
    """The result of a breadth first search.

    A SearchResult provides the ability to reconstruct the search or access a
    set of the keys the search found.
    """

    def __init__(self, start_keys, exclude_keys, key_count, keys):
        if False:
            print('Hello World!')
        'Create a SearchResult.\n\n        :param start_keys: The keys the search started at.\n        :param exclude_keys: The keys the search excludes.\n        :param key_count: The total number of keys (from start to but not\n            including exclude).\n        :param keys: The keys the search found. Note that in future we may get\n            a SearchResult from a smart server, in which case the keys list is\n            not necessarily immediately available.\n        '
        self._recipe = ('search', start_keys, exclude_keys, key_count)
        self._keys = frozenset(keys)

    def __repr__(self):
        if False:
            print('Hello World!')
        (kind, start_keys, exclude_keys, key_count) = self._recipe
        if len(start_keys) > 5:
            start_keys_repr = repr(list(start_keys)[:5])[:-1] + ', ...]'
        else:
            start_keys_repr = repr(start_keys)
        if len(exclude_keys) > 5:
            exclude_keys_repr = repr(list(exclude_keys)[:5])[:-1] + ', ...]'
        else:
            exclude_keys_repr = repr(exclude_keys)
        return '<%s %s:(%s, %s, %d)>' % (self.__class__.__name__, kind, start_keys_repr, exclude_keys_repr, key_count)

    def get_recipe(self):
        if False:
            return 10
        "Return a recipe that can be used to replay this search.\n\n        The recipe allows reconstruction of the same results at a later date\n        without knowing all the found keys. The essential elements are a list\n        of keys to start and to stop at. In order to give reproducible\n        results when ghosts are encountered by a search they are automatically\n        added to the exclude list (or else ghost filling may alter the\n        results).\n\n        :return: A tuple ('search', start_keys_set, exclude_keys_set,\n            revision_count). To recreate the results of this search, create a\n            breadth first searcher on the same graph starting at start_keys.\n            Then call next() (or next_with_ghosts()) repeatedly, and on every\n            result, call stop_searching_any on any keys from the exclude_keys\n            set. The revision_count value acts as a trivial cross-check - the\n            found revisions of the new search should have as many elements as\n            revision_count. If it does not, then additional revisions have been\n            ghosted since the search was executed the first time and the second\n            time.\n        "
        return self._recipe

    def get_network_struct(self):
        if False:
            return 10
        start_keys = ' '.join(self._recipe[1])
        stop_keys = ' '.join(self._recipe[2])
        count = str(self._recipe[3])
        return (self._recipe[0], '\n'.join((start_keys, stop_keys, count)))

    def get_keys(self):
        if False:
            print('Hello World!')
        'Return the keys found in this search.\n\n        :return: A set of keys.\n        '
        return self._keys

    def is_empty(self):
        if False:
            while True:
                i = 10
        'Return false if the search lists 1 or more revisions.'
        return self._recipe[3] == 0

    def refine(self, seen, referenced):
        if False:
            i = 10
            return i + 15
        'Create a new search by refining this search.\n\n        :param seen: Revisions that have been satisfied.\n        :param referenced: Revision references observed while satisfying some\n            of this search.\n        '
        start = self._recipe[1]
        exclude = self._recipe[2]
        count = self._recipe[3]
        keys = self.get_keys()
        pending_refs = set(referenced)
        pending_refs.update(start)
        pending_refs.difference_update(seen)
        pending_refs.difference_update(exclude)
        seen_heads = start.intersection(seen)
        exclude.update(seen_heads)
        keys = keys - seen
        count -= len(seen)
        return SearchResult(pending_refs, exclude, count, keys)

class PendingAncestryResult(AbstractSearchResult):
    """A search result that will reconstruct the ancestry for some graph heads.

    Unlike SearchResult, this doesn't hold the complete search result in
    memory, it just holds a description of how to generate it.
    """

    def __init__(self, heads, repo):
        if False:
            print('Hello World!')
        'Constructor.\n\n        :param heads: an iterable of graph heads.\n        :param repo: a repository to use to generate the ancestry for the given\n            heads.\n        '
        self.heads = frozenset(heads)
        self.repo = repo

    def __repr__(self):
        if False:
            while True:
                i = 10
        if len(self.heads) > 5:
            heads_repr = repr(list(self.heads)[:5])[:-1]
            heads_repr += ', <%d more>...]' % (len(self.heads) - 5,)
        else:
            heads_repr = repr(self.heads)
        return '<%s heads:%s repo:%r>' % (self.__class__.__name__, heads_repr, self.repo)

    def get_recipe(self):
        if False:
            for i in range(10):
                print('nop')
        "Return a recipe that can be used to replay this search.\n\n        The recipe allows reconstruction of the same results at a later date.\n\n        :seealso SearchResult.get_recipe:\n\n        :return: A tuple ('proxy-search', start_keys_set, set(), -1)\n            To recreate this result, create a PendingAncestryResult with the\n            start_keys_set.\n        "
        return ('proxy-search', self.heads, set(), -1)

    def get_network_struct(self):
        if False:
            print('Hello World!')
        parts = ['ancestry-of']
        parts.extend(self.heads)
        return parts

    def get_keys(self):
        if False:
            for i in range(10):
                print('nop')
        'See SearchResult.get_keys.\n\n        Returns all the keys for the ancestry of the heads, excluding\n        NULL_REVISION.\n        '
        return self._get_keys(self.repo.get_graph())

    def _get_keys(self, graph):
        if False:
            print('Hello World!')
        NULL_REVISION = revision.NULL_REVISION
        keys = [key for (key, parents) in graph.iter_ancestry(self.heads) if key != NULL_REVISION and parents is not None]
        return keys

    def is_empty(self):
        if False:
            print('Hello World!')
        'Return false if the search lists 1 or more revisions.'
        if revision.NULL_REVISION in self.heads:
            return len(self.heads) == 1
        else:
            return len(self.heads) == 0

    def refine(self, seen, referenced):
        if False:
            i = 10
            return i + 15
        'Create a new search by refining this search.\n\n        :param seen: Revisions that have been satisfied.\n        :param referenced: Revision references observed while satisfying some\n            of this search.\n        '
        referenced = self.heads.union(referenced)
        return PendingAncestryResult(referenced - seen, self.repo)

class EmptySearchResult(AbstractSearchResult):
    """An empty search result."""

    def is_empty(self):
        if False:
            print('Hello World!')
        return True

class EverythingResult(AbstractSearchResult):
    """A search result that simply requests everything in the repository."""

    def __init__(self, repo):
        if False:
            print('Hello World!')
        self._repo = repo

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s(%r)' % (self.__class__.__name__, self._repo)

    def get_recipe(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError(self.get_recipe)

    def get_network_struct(self):
        if False:
            i = 10
            return i + 15
        return ('everything',)

    def get_keys(self):
        if False:
            while True:
                i = 10
        if 'evil' in debug.debug_flags:
            from bzrlib import remote
            if isinstance(self._repo, remote.RemoteRepository):
                trace.mutter_callsite(2, 'EverythingResult(RemoteRepository).get_keys() is slow.')
        return self._repo.all_revision_ids()

    def is_empty(self):
        if False:
            print('Hello World!')
        return False

    def refine(self, seen, referenced):
        if False:
            for i in range(10):
                print('nop')
        heads = set(self._repo.all_revision_ids())
        heads.difference_update(seen)
        heads.update(referenced)
        return PendingAncestryResult(heads, self._repo)

class EverythingNotInOther(AbstractSearch):
    """Find all revisions in that are in one repo but not the other."""

    def __init__(self, to_repo, from_repo, find_ghosts=False):
        if False:
            i = 10
            return i + 15
        self.to_repo = to_repo
        self.from_repo = from_repo
        self.find_ghosts = find_ghosts

    def execute(self):
        if False:
            for i in range(10):
                print('nop')
        return self.to_repo.search_missing_revision_ids(self.from_repo, find_ghosts=self.find_ghosts)

class NotInOtherForRevs(AbstractSearch):
    """Find all revisions missing in one repo for a some specific heads."""

    def __init__(self, to_repo, from_repo, required_ids, if_present_ids=None, find_ghosts=False, limit=None):
        if False:
            print('Hello World!')
        'Constructor.\n\n        :param required_ids: revision IDs of heads that must be found, or else\n            the search will fail with NoSuchRevision.  All revisions in their\n            ancestry not already in the other repository will be included in\n            the search result.\n        :param if_present_ids: revision IDs of heads that may be absent in the\n            source repository.  If present, then their ancestry not already\n            found in other will be included in the search result.\n        :param limit: maximum number of revisions to fetch\n        '
        self.to_repo = to_repo
        self.from_repo = from_repo
        self.find_ghosts = find_ghosts
        self.required_ids = required_ids
        self.if_present_ids = if_present_ids
        self.limit = limit

    def __repr__(self):
        if False:
            while True:
                i = 10
        if len(self.required_ids) > 5:
            reqd_revs_repr = repr(list(self.required_ids)[:5])[:-1] + ', ...]'
        else:
            reqd_revs_repr = repr(self.required_ids)
        if self.if_present_ids and len(self.if_present_ids) > 5:
            ifp_revs_repr = repr(list(self.if_present_ids)[:5])[:-1] + ', ...]'
        else:
            ifp_revs_repr = repr(self.if_present_ids)
        return "<%s from:%r to:%r find_ghosts:%r req'd:%r if-present:%rlimit:%r>" % (self.__class__.__name__, self.from_repo, self.to_repo, self.find_ghosts, reqd_revs_repr, ifp_revs_repr, self.limit)

    def execute(self):
        if False:
            while True:
                i = 10
        return self.to_repo.search_missing_revision_ids(self.from_repo, revision_ids=self.required_ids, if_present_ids=self.if_present_ids, find_ghosts=self.find_ghosts, limit=self.limit)

def search_result_from_parent_map(parent_map, missing_keys):
    if False:
        return 10
    'Transform a parent_map into SearchResult information.'
    if not parent_map:
        return ([], [], 0)
    start_set = set(parent_map)
    result_parents = set()
    for parents in parent_map.itervalues():
        result_parents.update(parents)
    stop_keys = result_parents.difference(start_set)
    stop_keys.difference_update(missing_keys)
    key_count = len(parent_map)
    if revision.NULL_REVISION in result_parents and revision.NULL_REVISION in missing_keys:
        key_count += 1
    included_keys = start_set.intersection(result_parents)
    start_set.difference_update(included_keys)
    return (start_set, stop_keys, key_count)

def _run_search(parent_map, heads, exclude_keys):
    if False:
        return 10
    'Given a parent map, run a _BreadthFirstSearcher on it.\n\n    Start at heads, walk until you hit exclude_keys. As a further improvement,\n    watch for any heads that you encounter while walking, which means they were\n    not heads of the search.\n\n    This is mostly used to generate a succinct recipe for how to walk through\n    most of parent_map.\n\n    :return: (_BreadthFirstSearcher, set(heads_encountered_by_walking))\n    '
    g = Graph(DictParentsProvider(parent_map))
    s = g._make_breadth_first_searcher(heads)
    found_heads = set()
    while True:
        try:
            next_revs = s.next()
        except StopIteration:
            break
        for parents in s._current_parents.itervalues():
            f_heads = heads.intersection(parents)
            if f_heads:
                found_heads.update(f_heads)
        stop_keys = exclude_keys.intersection(next_revs)
        if stop_keys:
            s.stop_searching_any(stop_keys)
    for parents in s._current_parents.itervalues():
        f_heads = heads.intersection(parents)
        if f_heads:
            found_heads.update(f_heads)
    return (s, found_heads)

def _find_possible_heads(parent_map, tip_keys, depth):
    if False:
        return 10
    "Walk backwards (towards children) through the parent_map.\n\n    This finds 'heads' that will hopefully succinctly describe our search\n    graph.\n    "
    child_map = invert_parent_map(parent_map)
    heads = set()
    current_roots = tip_keys
    walked = set(current_roots)
    while current_roots and depth > 0:
        depth -= 1
        children = set()
        children_update = children.update
        for p in current_roots:
            try:
                children_update(child_map[p])
            except KeyError:
                heads.add(p)
        children = children.difference(walked)
        walked.update(children)
        current_roots = children
    if current_roots:
        heads.update(current_roots)
    return heads

def limited_search_result_from_parent_map(parent_map, missing_keys, tip_keys, depth):
    if False:
        while True:
            i = 10
    "Transform a parent_map that is searching 'tip_keys' into an\n    approximate SearchResult.\n\n    We should be able to generate a SearchResult from a given set of starting\n    keys, that covers a subset of parent_map that has the last step pointing at\n    tip_keys. This is to handle the case that really-long-searches shouldn't be\n    started from scratch on each get_parent_map request, but we *do* want to\n    filter out some of the keys that we've already seen, so we don't get\n    information that we already know about on every request.\n\n    The server will validate the search (that starting at start_keys and\n    stopping at stop_keys yields the exact key_count), so we have to be careful\n    to give an exact recipe.\n\n    Basic algorithm is:\n        1) Invert parent_map to get child_map (todo: have it cached and pass it\n           in)\n        2) Starting at tip_keys, walk towards children for 'depth' steps.\n        3) At that point, we have the 'start' keys.\n        4) Start walking parent_map from 'start' keys, counting how many keys\n           are seen, and generating stop_keys for anything that would walk\n           outside of the parent_map.\n\n    :param parent_map: A map from {child_id: (parent_ids,)}\n    :param missing_keys: parent_ids that we know are unavailable\n    :param tip_keys: the revision_ids that we are searching\n    :param depth: How far back to walk.\n    "
    if not parent_map:
        return ([], [], 0)
    heads = _find_possible_heads(parent_map, tip_keys, depth)
    (s, found_heads) = _run_search(parent_map, heads, set(tip_keys))
    (start_keys, exclude_keys, keys) = s.get_state()
    if found_heads:
        start_keys = set(start_keys).difference(found_heads)
    return (start_keys, exclude_keys, len(keys))