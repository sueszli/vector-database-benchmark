from __future__ import absolute_import
from bzrlib.lazy_import import lazy_import
lazy_import(globals(), '\nfrom bzrlib import patiencediff\n')

class TextMerge(object):
    """Base class for text-mergers
    Subclasses must implement _merge_struct.

    Many methods produce or consume structured merge information.
    This is an iterable of tuples of lists of lines.
    Each tuple may have a length of 1 - 3, depending on whether the region it
    represents is conflicted.

    Unconflicted region tuples have length 1.
    Conflicted region tuples have length 2 or 3.  Index 1 is text_a, e.g. THIS.
    Index 1 is text_b, e.g. OTHER.  Index 2 is optional.  If present, it
    represents BASE.
    """
    A_MARKER = '<<<<<<< \n'
    B_MARKER = '>>>>>>> \n'
    SPLIT_MARKER = '=======\n'

    def __init__(self, a_marker=A_MARKER, b_marker=B_MARKER, split_marker=SPLIT_MARKER):
        if False:
            print('Hello World!')
        self.a_marker = a_marker
        self.b_marker = b_marker
        self.split_marker = split_marker

    def _merge_struct(self):
        if False:
            while True:
                i = 10
        'Return structured merge info.  Must be implemented by subclasses.\n        See TextMerge docstring for details on the format.\n        '
        raise NotImplementedError('_merge_struct is abstract')

    def struct_to_lines(self, struct_iter):
        if False:
            print('Hello World!')
        'Convert merge result tuples to lines'
        for lines in struct_iter:
            if len(lines) == 1:
                for line in lines[0]:
                    yield line
            else:
                yield self.a_marker
                for line in lines[0]:
                    yield line
                yield self.split_marker
                for line in lines[1]:
                    yield line
                yield self.b_marker

    def iter_useful(self, struct_iter):
        if False:
            for i in range(10):
                print('nop')
        'Iterate through input tuples, skipping empty ones.'
        for group in struct_iter:
            if len(group[0]) > 0:
                yield group
            elif len(group) > 1 and len(group[1]) > 0:
                yield group

    def merge_lines(self, reprocess=False):
        if False:
            for i in range(10):
                print('nop')
        'Produce an iterable of lines, suitable for writing to a file\n        Returns a tuple of (line iterable, conflict indicator)\n        If reprocess is True, a two-way merge will be performed on the\n        intermediate structure, to reduce conflict regions.\n        '
        struct = []
        conflicts = False
        for group in self.merge_struct(reprocess):
            struct.append(group)
            if len(group) > 1:
                conflicts = True
        return (self.struct_to_lines(struct), conflicts)

    def merge_struct(self, reprocess=False):
        if False:
            i = 10
            return i + 15
        'Produce structured merge info'
        struct_iter = self.iter_useful(self._merge_struct())
        if reprocess is True:
            return self.reprocess_struct(struct_iter)
        else:
            return struct_iter

    @staticmethod
    def reprocess_struct(struct_iter):
        if False:
            i = 10
            return i + 15
        ' Perform a two-way merge on structural merge info.\n        This reduces the size of conflict regions, but breaks the connection\n        between the BASE text and the conflict region.\n\n        This process may split a single conflict region into several smaller\n        ones, but will not introduce new conflicts.\n        '
        for group in struct_iter:
            if len(group) == 1:
                yield group
            else:
                for newgroup in Merge2(group[0], group[1]).merge_struct():
                    yield newgroup

class Merge2(TextMerge):
    """ Two-way merge.
    In a two way merge, common regions are shown as unconflicting, and uncommon
    regions produce conflicts.
    """

    def __init__(self, lines_a, lines_b, a_marker=TextMerge.A_MARKER, b_marker=TextMerge.B_MARKER, split_marker=TextMerge.SPLIT_MARKER):
        if False:
            while True:
                i = 10
        TextMerge.__init__(self, a_marker, b_marker, split_marker)
        self.lines_a = lines_a
        self.lines_b = lines_b

    def _merge_struct(self):
        if False:
            while True:
                i = 10
        'Return structured merge info.\n        See TextMerge docstring.\n        '
        sm = patiencediff.PatienceSequenceMatcher(None, self.lines_a, self.lines_b)
        pos_a = 0
        pos_b = 0
        for (ai, bi, l) in sm.get_matching_blocks():
            yield (self.lines_a[pos_a:ai], self.lines_b[pos_b:bi])
            yield (self.lines_a[ai:ai + l],)
            pos_a = ai + l
            pos_b = bi + l
        yield (self.lines_a[pos_a:-1], self.lines_b[pos_b:-1])