"""Merge logic for changelog_merge plugin."""
from __future__ import absolute_import
import difflib
from bzrlib import debug, merge, urlutils
from bzrlib.merge3 import Merge3
from bzrlib.trace import mutter

def changelog_entries(lines):
    if False:
        for i in range(10):
            print('nop')
    'Return a list of changelog entries.\n\n    :param lines: lines of a changelog file.\n    :returns: list of entries.  Each entry is a tuple of lines.\n    '
    entries = []
    for line in lines:
        if line[0] not in (' ', '\t', '\n'):
            entries.append([line])
        else:
            try:
                entry = entries[-1]
            except IndexError:
                entries.append([])
                entry = entries[-1]
            entry.append(line)
    return map(tuple, entries)

def entries_to_lines(entries):
    if False:
        i = 10
        return i + 15
    'Turn a list of entries into a flat iterable of lines.'
    for entry in entries:
        for line in entry:
            yield line

class ChangeLogMerger(merge.ConfigurableFileMerger):
    """Merge GNU-format ChangeLog files."""
    name_prefix = 'changelog'

    def get_filepath(self, params, tree):
        if False:
            return 10
        'Calculate the path to the file in a tree.\n\n        This is overridden to return just the basename, rather than full path,\n        so that e.g. if the config says ``changelog_merge_files = ChangeLog``,\n        then all ChangeLog files in the tree will match (not just one in the\n        root of the tree).\n        \n        :param params: A MergeHookParams describing the file to merge\n        :param tree: a Tree, e.g. self.merger.this_tree.\n        '
        return urlutils.basename(tree.id2path(params.file_id))

    def merge_text(self, params):
        if False:
            i = 10
            return i + 15
        'Merge changelog changes.\n\n         * new entries from other will float to the top\n         * edits to older entries are preserved\n        '
        this_entries = changelog_entries(params.this_lines)
        other_entries = changelog_entries(params.other_lines)
        base_entries = changelog_entries(params.base_lines)
        try:
            result_entries = merge_entries(base_entries, this_entries, other_entries)
        except EntryConflict:
            return ('not_applicable', None)
        return ('success', entries_to_lines(result_entries))

class EntryConflict(Exception):
    pass

def default_guess_edits(new_entries, deleted_entries, entry_as_str=''.join):
    if False:
        for i in range(10):
            print('nop')
    "Default implementation of guess_edits param of merge_entries.\n\n    This algorithm does O(N^2 * logN) SequenceMatcher.ratio() calls, which is\n    pretty bad, but it shouldn't be used very often.\n    "
    deleted_entries_as_strs = map(entry_as_str, deleted_entries)
    new_entries_as_strs = map(entry_as_str, new_entries)
    result_new = list(new_entries)
    result_deleted = list(deleted_entries)
    result_edits = []
    sm = difflib.SequenceMatcher()
    CUTOFF = 0.8
    while True:
        best = None
        best_score = CUTOFF
        for new_entry_as_str in new_entries_as_strs:
            sm.set_seq1(new_entry_as_str)
            for old_entry_as_str in deleted_entries_as_strs:
                sm.set_seq2(old_entry_as_str)
                score = sm.ratio()
                if score > best_score:
                    best = (new_entry_as_str, old_entry_as_str)
                    best_score = score
        if best is not None:
            del_index = deleted_entries_as_strs.index(best[1])
            new_index = new_entries_as_strs.index(best[0])
            result_edits.append((result_deleted[del_index], result_new[new_index]))
            del deleted_entries_as_strs[del_index], result_deleted[del_index]
            del new_entries_as_strs[new_index], result_new[new_index]
        else:
            break
    return (result_new, result_deleted, result_edits)

def merge_entries(base_entries, this_entries, other_entries, guess_edits=default_guess_edits):
    if False:
        while True:
            i = 10
    'Merge changelog given base, this, and other versions.'
    m3 = Merge3(base_entries, this_entries, other_entries, allow_objects=True)
    result_entries = []
    at_top = True
    for group in m3.merge_groups():
        if 'changelog_merge' in debug.debug_flags:
            mutter('merge group:\n%r', group)
        group_kind = group[0]
        if group_kind == 'conflict':
            (_, base, this, other) = group
            new_in_other = [entry for entry in other if entry not in base]
            deleted_in_other = [entry for entry in base if entry not in other]
            if at_top and deleted_in_other:
                (new_in_other, deleted_in_other, edits_in_other) = guess_edits(new_in_other, deleted_in_other)
            else:
                edits_in_other = []
            if 'changelog_merge' in debug.debug_flags:
                mutter('at_top: %r', at_top)
                mutter('new_in_other: %r', new_in_other)
                mutter('deleted_in_other: %r', deleted_in_other)
                mutter('edits_in_other: %r', edits_in_other)
            updated_this = [entry for entry in this if entry not in deleted_in_other]
            for (old_entry, new_entry) in edits_in_other:
                try:
                    index = updated_this.index(old_entry)
                except ValueError:
                    raise EntryConflict()
                updated_this[index] = new_entry
            if 'changelog_merge' in debug.debug_flags:
                mutter('updated_this: %r', updated_this)
            if at_top:
                result_entries = new_in_other + result_entries
            else:
                result_entries.extend(new_in_other)
            result_entries.extend(updated_this)
        else:
            lines = group[1]
            result_entries.extend(lines)
        at_top = False
    return result_entries