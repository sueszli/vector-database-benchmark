"""Merge logic for news_merge plugin."""
from __future__ import absolute_import
from bzrlib.plugins.news_merge.parser import simple_parse_lines
from bzrlib import merge, merge3

class NewsMerger(merge.ConfigurableFileMerger):
    """Merge bzr NEWS files."""
    name_prefix = 'news'

    def merge_text(self, params):
        if False:
            return 10
        'Perform a simple 3-way merge of a bzr NEWS file.\n\n        Each section of a bzr NEWS file is essentially an ordered set of bullet\n        points, so we can simply take a set of bullet points, determine which\n        bullets to add and which to remove, sort, and reserialize.\n        '
        this_lines = list(simple_parse_lines(params.this_lines))
        other_lines = list(simple_parse_lines(params.other_lines))
        base_lines = list(simple_parse_lines(params.base_lines))
        m3 = merge3.Merge3(base_lines, this_lines, other_lines, allow_objects=True)
        result_chunks = []
        for group in m3.merge_groups():
            if group[0] == 'conflict':
                (_, base, a, b) = group
                for line_set in [base, a, b]:
                    for line in line_set:
                        if line[0] != 'bullet':
                            return ('not_applicable', None)
                new_in_a = set(a).difference(base)
                new_in_b = set(b).difference(base)
                all_new = new_in_a.union(new_in_b)
                deleted_in_a = set(base).difference(a)
                deleted_in_b = set(base).difference(b)
                final = all_new.difference(deleted_in_a).difference(deleted_in_b)
                final = sorted(final, key=sort_key)
                result_chunks.extend(final)
            else:
                result_chunks.extend(group[1])
        result_lines = '\n\n'.join((chunk[1] for chunk in result_chunks))
        return ('success', result_lines)

def sort_key(chunk):
    if False:
        return 10
    return chunk[1].replace('`', '').lower()