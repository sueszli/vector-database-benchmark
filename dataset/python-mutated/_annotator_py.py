"""Functionality for doing annotations in the 'optimal' way"""
from __future__ import absolute_import
from bzrlib.lazy_import import lazy_import
lazy_import(globals(), '\nfrom bzrlib import (\n    annotate, # Must be lazy to avoid circular importing\n    graph as _mod_graph,\n    patiencediff,\n    )\n')
from bzrlib import errors, osutils, ui

class Annotator(object):
    """Class that drives performing annotations."""

    def __init__(self, vf):
        if False:
            for i in range(10):
                print('nop')
        'Create a new Annotator from a VersionedFile.'
        self._vf = vf
        self._parent_map = {}
        self._text_cache = {}
        self._num_needed_children = {}
        self._annotations_cache = {}
        self._heads_provider = None
        self._ann_tuple_cache = {}

    def _update_needed_children(self, key, parent_keys):
        if False:
            for i in range(10):
                print('nop')
        for parent_key in parent_keys:
            if parent_key in self._num_needed_children:
                self._num_needed_children[parent_key] += 1
            else:
                self._num_needed_children[parent_key] = 1

    def _get_needed_keys(self, key):
        if False:
            for i in range(10):
                print('nop')
        "Determine the texts we need to get from the backing vf.\n\n        :return: (vf_keys_needed, ann_keys_needed)\n            vf_keys_needed  These are keys that we need to get from the vf\n            ann_keys_needed Texts which we have in self._text_cache but we\n                            don't have annotations for. We need to yield these\n                            in the proper order so that we can get proper\n                            annotations.\n        "
        parent_map = self._parent_map
        self._num_needed_children[key] = 1
        vf_keys_needed = set()
        ann_keys_needed = set()
        needed_keys = set([key])
        while needed_keys:
            parent_lookup = []
            next_parent_map = {}
            for key in needed_keys:
                if key in self._parent_map:
                    if key not in self._text_cache:
                        vf_keys_needed.add(key)
                    elif key not in self._annotations_cache:
                        ann_keys_needed.add(key)
                        next_parent_map[key] = self._parent_map[key]
                else:
                    parent_lookup.append(key)
                    vf_keys_needed.add(key)
            needed_keys = set()
            next_parent_map.update(self._vf.get_parent_map(parent_lookup))
            for (key, parent_keys) in next_parent_map.iteritems():
                if parent_keys is None:
                    parent_keys = ()
                    next_parent_map[key] = ()
                self._update_needed_children(key, parent_keys)
                needed_keys.update([key for key in parent_keys if key not in parent_map])
            parent_map.update(next_parent_map)
            self._heads_provider = None
        return (vf_keys_needed, ann_keys_needed)

    def _get_needed_texts(self, key, pb=None):
        if False:
            while True:
                i = 10
        "Get the texts we need to properly annotate key.\n\n        :param key: A Key that is present in self._vf\n        :return: Yield (this_key, text, num_lines)\n            'text' is an opaque object that just has to work with whatever\n            matcher object we are using. Currently it is always 'lines' but\n            future improvements may change this to a simple text string.\n        "
        (keys, ann_keys) = self._get_needed_keys(key)
        if pb is not None:
            pb.update('getting stream', 0, len(keys))
        stream = self._vf.get_record_stream(keys, 'topological', True)
        for (idx, record) in enumerate(stream):
            if pb is not None:
                pb.update('extracting', 0, len(keys))
            if record.storage_kind == 'absent':
                raise errors.RevisionNotPresent(record.key, self._vf)
            this_key = record.key
            lines = osutils.chunks_to_lines(record.get_bytes_as('chunked'))
            num_lines = len(lines)
            self._text_cache[this_key] = lines
            yield (this_key, lines, num_lines)
        for key in ann_keys:
            lines = self._text_cache[key]
            num_lines = len(lines)
            yield (key, lines, num_lines)

    def _get_parent_annotations_and_matches(self, key, text, parent_key):
        if False:
            while True:
                i = 10
        'Get the list of annotations for the parent, and the matching lines.\n\n        :param text: The opaque value given by _get_needed_texts\n        :param parent_key: The key for the parent text\n        :return: (parent_annotations, matching_blocks)\n            parent_annotations is a list as long as the number of lines in\n                parent\n            matching_blocks is a list of (parent_idx, text_idx, len) tuples\n                indicating which lines match between the two texts\n        '
        parent_lines = self._text_cache[parent_key]
        parent_annotations = self._annotations_cache[parent_key]
        matcher = patiencediff.PatienceSequenceMatcher(None, parent_lines, text)
        matching_blocks = matcher.get_matching_blocks()
        return (parent_annotations, matching_blocks)

    def _update_from_first_parent(self, key, annotations, lines, parent_key):
        if False:
            for i in range(10):
                print('nop')
        'Reannotate this text relative to its first parent.'
        (parent_annotations, matching_blocks) = self._get_parent_annotations_and_matches(key, lines, parent_key)
        for (parent_idx, lines_idx, match_len) in matching_blocks:
            annotations[lines_idx:lines_idx + match_len] = parent_annotations[parent_idx:parent_idx + match_len]

    def _update_from_other_parents(self, key, annotations, lines, this_annotation, parent_key):
        if False:
            return 10
        'Reannotate this text relative to a second (or more) parent.'
        (parent_annotations, matching_blocks) = self._get_parent_annotations_and_matches(key, lines, parent_key)
        last_ann = None
        last_parent = None
        last_res = None
        for (parent_idx, lines_idx, match_len) in matching_blocks:
            ann_sub = annotations[lines_idx:lines_idx + match_len]
            par_sub = parent_annotations[parent_idx:parent_idx + match_len]
            if ann_sub == par_sub:
                continue
            for idx in xrange(match_len):
                ann = ann_sub[idx]
                par_ann = par_sub[idx]
                ann_idx = lines_idx + idx
                if ann == par_ann:
                    continue
                if ann == this_annotation:
                    annotations[ann_idx] = par_ann
                    continue
                if ann == last_ann and par_ann == last_parent:
                    annotations[ann_idx] = last_res
                else:
                    new_ann = set(ann)
                    new_ann.update(par_ann)
                    new_ann = tuple(sorted(new_ann))
                    annotations[ann_idx] = new_ann
                    last_ann = ann
                    last_parent = par_ann
                    last_res = new_ann

    def _record_annotation(self, key, parent_keys, annotations):
        if False:
            while True:
                i = 10
        self._annotations_cache[key] = annotations
        for parent_key in parent_keys:
            num = self._num_needed_children[parent_key]
            num -= 1
            if num == 0:
                del self._text_cache[parent_key]
                del self._annotations_cache[parent_key]
            self._num_needed_children[parent_key] = num

    def _annotate_one(self, key, text, num_lines):
        if False:
            for i in range(10):
                print('nop')
        this_annotation = (key,)
        annotations = [this_annotation] * num_lines
        parent_keys = self._parent_map[key]
        if parent_keys:
            self._update_from_first_parent(key, annotations, text, parent_keys[0])
            for parent in parent_keys[1:]:
                self._update_from_other_parents(key, annotations, text, this_annotation, parent)
        self._record_annotation(key, parent_keys, annotations)

    def add_special_text(self, key, parent_keys, text):
        if False:
            for i in range(10):
                print('nop')
        "Add a specific text to the graph.\n\n        This is used to add a text which is not otherwise present in the\n        versioned file. (eg. a WorkingTree injecting 'current:' into the\n        graph to annotate the edited content.)\n\n        :param key: The key to use to request this text be annotated\n        :param parent_keys: The parents of this text\n        :param text: A string containing the content of the text\n        "
        self._parent_map[key] = parent_keys
        self._text_cache[key] = osutils.split_lines(text)
        self._heads_provider = None

    def annotate(self, key):
        if False:
            i = 10
            return i + 15
        'Return annotated fulltext for the given key.\n\n        :param key: A tuple defining the text to annotate\n        :return: ([annotations], [lines])\n            annotations is a list of tuples of keys, one for each line in lines\n                        each key is a possible source for the given line.\n            lines the text of "key" as a list of lines\n        '
        pb = ui.ui_factory.nested_progress_bar()
        try:
            for (text_key, text, num_lines) in self._get_needed_texts(key, pb=pb):
                self._annotate_one(text_key, text, num_lines)
        finally:
            pb.finished()
        try:
            annotations = self._annotations_cache[key]
        except KeyError:
            raise errors.RevisionNotPresent(key, self._vf)
        return (annotations, self._text_cache[key])

    def _get_heads_provider(self):
        if False:
            return 10
        if self._heads_provider is None:
            self._heads_provider = _mod_graph.KnownGraph(self._parent_map)
        return self._heads_provider

    def _resolve_annotation_tie(self, the_heads, line, tiebreaker):
        if False:
            i = 10
            return i + 15
        if tiebreaker is None:
            head = sorted(the_heads)[0]
        else:
            next_head = iter(the_heads)
            head = next_head.next()
            for possible_head in next_head:
                annotated_lines = ((head, line), (possible_head, line))
                head = tiebreaker(annotated_lines)[0]
        return head

    def annotate_flat(self, key):
        if False:
            i = 10
            return i + 15
        'Determine the single-best-revision to source for each line.\n\n        This is meant as a compatibility thunk to how annotate() used to work.\n        :return: [(ann_key, line)]\n            A list of tuples with a single annotation key for each line.\n        '
        custom_tiebreaker = annotate._break_annotation_tie
        (annotations, lines) = self.annotate(key)
        out = []
        heads = self._get_heads_provider().heads
        append = out.append
        for (annotation, line) in zip(annotations, lines):
            if len(annotation) == 1:
                head = annotation[0]
            else:
                the_heads = heads(annotation)
                if len(the_heads) == 1:
                    for head in the_heads:
                        break
                else:
                    head = self._resolve_annotation_tie(the_heads, line, custom_tiebreaker)
            append((head, line))
        return out