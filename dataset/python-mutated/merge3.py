from __future__ import absolute_import
from bzrlib import errors, patiencediff, textfile

def intersect(ra, rb):
    if False:
        for i in range(10):
            print('nop')
    'Given two ranges return the range where they intersect or None.\n\n    >>> intersect((0, 10), (0, 6))\n    (0, 6)\n    >>> intersect((0, 10), (5, 15))\n    (5, 10)\n    >>> intersect((0, 10), (10, 15))\n    >>> intersect((0, 9), (10, 15))\n    >>> intersect((0, 9), (7, 15))\n    (7, 9)\n    '
    sa = max(ra[0], rb[0])
    sb = min(ra[1], rb[1])
    if sa < sb:
        return (sa, sb)
    else:
        return None

def compare_range(a, astart, aend, b, bstart, bend):
    if False:
        return 10
    'Compare a[astart:aend] == b[bstart:bend], without slicing.\n    '
    if aend - astart != bend - bstart:
        return False
    for (ia, ib) in zip(xrange(astart, aend), xrange(bstart, bend)):
        if a[ia] != b[ib]:
            return False
    else:
        return True

class Merge3(object):
    """3-way merge of texts.

    Given BASE, OTHER, THIS, tries to produce a combined text
    incorporating the changes from both BASE->OTHER and BASE->THIS.
    All three will typically be sequences of lines."""

    def __init__(self, base, a, b, is_cherrypick=False, allow_objects=False):
        if False:
            print('Hello World!')
        'Constructor.\n\n        :param base: lines in BASE\n        :param a: lines in A\n        :param b: lines in B\n        :param is_cherrypick: flag indicating if this merge is a cherrypick.\n            When cherrypicking b => a, matches with b and base do not conflict.\n        :param allow_objects: if True, do not require that base, a and b are\n            plain Python strs.  Also prevents BinaryFile from being raised.\n            Lines can be any sequence of comparable and hashable Python\n            objects.\n        '
        if not allow_objects:
            textfile.check_text_lines(base)
            textfile.check_text_lines(a)
            textfile.check_text_lines(b)
        self.base = base
        self.a = a
        self.b = b
        self.is_cherrypick = is_cherrypick

    def merge_lines(self, name_a=None, name_b=None, name_base=None, start_marker='<<<<<<<', mid_marker='=======', end_marker='>>>>>>>', base_marker=None, reprocess=False):
        if False:
            return 10
        'Return merge in cvs-like form.\n        '
        newline = '\n'
        if len(self.a) > 0:
            if self.a[0].endswith('\r\n'):
                newline = '\r\n'
            elif self.a[0].endswith('\r'):
                newline = '\r'
        if base_marker and reprocess:
            raise errors.CantReprocessAndShowBase()
        if name_a:
            start_marker = start_marker + ' ' + name_a
        if name_b:
            end_marker = end_marker + ' ' + name_b
        if name_base and base_marker:
            base_marker = base_marker + ' ' + name_base
        merge_regions = self.merge_regions()
        if reprocess is True:
            merge_regions = self.reprocess_merge_regions(merge_regions)
        for t in merge_regions:
            what = t[0]
            if what == 'unchanged':
                for i in range(t[1], t[2]):
                    yield self.base[i]
            elif what == 'a' or what == 'same':
                for i in range(t[1], t[2]):
                    yield self.a[i]
            elif what == 'b':
                for i in range(t[1], t[2]):
                    yield self.b[i]
            elif what == 'conflict':
                yield (start_marker + newline)
                for i in range(t[3], t[4]):
                    yield self.a[i]
                if base_marker is not None:
                    yield (base_marker + newline)
                    for i in range(t[1], t[2]):
                        yield self.base[i]
                yield (mid_marker + newline)
                for i in range(t[5], t[6]):
                    yield self.b[i]
                yield (end_marker + newline)
            else:
                raise ValueError(what)

    def merge_annotated(self):
        if False:
            for i in range(10):
                print('nop')
        'Return merge with conflicts, showing origin of lines.\n\n        Most useful for debugging merge.\n        '
        for t in self.merge_regions():
            what = t[0]
            if what == 'unchanged':
                for i in range(t[1], t[2]):
                    yield ('u | ' + self.base[i])
            elif what == 'a' or what == 'same':
                for i in range(t[1], t[2]):
                    yield (what[0] + ' | ' + self.a[i])
            elif what == 'b':
                for i in range(t[1], t[2]):
                    yield ('b | ' + self.b[i])
            elif what == 'conflict':
                yield '<<<<\n'
                for i in range(t[3], t[4]):
                    yield ('A | ' + self.a[i])
                yield '----\n'
                for i in range(t[5], t[6]):
                    yield ('B | ' + self.b[i])
                yield '>>>>\n'
            else:
                raise ValueError(what)

    def merge_groups(self):
        if False:
            print('Hello World!')
        "Yield sequence of line groups.  Each one is a tuple:\n\n        'unchanged', lines\n             Lines unchanged from base\n\n        'a', lines\n             Lines taken from a\n\n        'same', lines\n             Lines taken from a (and equal to b)\n\n        'b', lines\n             Lines taken from b\n\n        'conflict', base_lines, a_lines, b_lines\n             Lines from base were changed to either a or b and conflict.\n        "
        for t in self.merge_regions():
            what = t[0]
            if what == 'unchanged':
                yield (what, self.base[t[1]:t[2]])
            elif what == 'a' or what == 'same':
                yield (what, self.a[t[1]:t[2]])
            elif what == 'b':
                yield (what, self.b[t[1]:t[2]])
            elif what == 'conflict':
                yield (what, self.base[t[1]:t[2]], self.a[t[3]:t[4]], self.b[t[5]:t[6]])
            else:
                raise ValueError(what)

    def merge_regions(self):
        if False:
            for i in range(10):
                print('nop')
        'Return sequences of matching and conflicting regions.\n\n        This returns tuples, where the first value says what kind we\n        have:\n\n        \'unchanged\', start, end\n             Take a region of base[start:end]\n\n        \'same\', astart, aend\n             b and a are different from base but give the same result\n\n        \'a\', start, end\n             Non-clashing insertion from a[start:end]\n\n        Method is as follows:\n\n        The two sequences align only on regions which match the base\n        and both descendents.  These are found by doing a two-way diff\n        of each one against the base, and then finding the\n        intersections between those regions.  These "sync regions"\n        are by definition unchanged in both and easily dealt with.\n\n        The regions in between can be in any of three cases:\n        conflicted, or changed on only one side.\n        '
        iz = ia = ib = 0
        for (zmatch, zend, amatch, aend, bmatch, bend) in self.find_sync_regions():
            matchlen = zend - zmatch
            len_a = amatch - ia
            len_b = bmatch - ib
            len_base = zmatch - iz
            if len_a or len_b:
                same = compare_range(self.a, ia, amatch, self.b, ib, bmatch)
                if same:
                    yield ('same', ia, amatch)
                else:
                    equal_a = compare_range(self.a, ia, amatch, self.base, iz, zmatch)
                    equal_b = compare_range(self.b, ib, bmatch, self.base, iz, zmatch)
                    if equal_a and (not equal_b):
                        yield ('b', ib, bmatch)
                    elif equal_b and (not equal_a):
                        yield ('a', ia, amatch)
                    elif not equal_a and (not equal_b):
                        if self.is_cherrypick:
                            for node in self._refine_cherrypick_conflict(iz, zmatch, ia, amatch, ib, bmatch):
                                yield node
                        else:
                            yield ('conflict', iz, zmatch, ia, amatch, ib, bmatch)
                    else:
                        raise AssertionError("can't handle a=b=base but unmatched")
                ia = amatch
                ib = bmatch
            iz = zmatch
            if matchlen > 0:
                yield ('unchanged', zmatch, zend)
                iz = zend
                ia = aend
                ib = bend

    def _refine_cherrypick_conflict(self, zstart, zend, astart, aend, bstart, bend):
        if False:
            return 10
        'When cherrypicking b => a, ignore matches with b and base.'
        matches = patiencediff.PatienceSequenceMatcher(None, self.base[zstart:zend], self.b[bstart:bend]).get_matching_blocks()
        last_base_idx = 0
        last_b_idx = 0
        last_b_idx = 0
        yielded_a = False
        for (base_idx, b_idx, match_len) in matches:
            conflict_z_len = base_idx - last_base_idx
            conflict_b_len = b_idx - last_b_idx
            if conflict_b_len == 0:
                pass
            elif yielded_a:
                yield ('conflict', zstart + last_base_idx, zstart + base_idx, aend, aend, bstart + last_b_idx, bstart + b_idx)
            else:
                yielded_a = True
                yield ('conflict', zstart + last_base_idx, zstart + base_idx, astart, aend, bstart + last_b_idx, bstart + b_idx)
            last_base_idx = base_idx + match_len
            last_b_idx = b_idx + match_len
        if last_base_idx != zend - zstart or last_b_idx != bend - bstart:
            if yielded_a:
                yield ('conflict', zstart + last_base_idx, zstart + base_idx, aend, aend, bstart + last_b_idx, bstart + b_idx)
            else:
                yielded_a = True
                yield ('conflict', zstart + last_base_idx, zstart + base_idx, astart, aend, bstart + last_b_idx, bstart + b_idx)
        if not yielded_a:
            yield ('conflict', zstart, zend, astart, aend, bstart, bend)

    def reprocess_merge_regions(self, merge_regions):
        if False:
            while True:
                i = 10
        'Where there are conflict regions, remove the agreed lines.\n\n        Lines where both A and B have made the same changes are\n        eliminated.\n        '
        for region in merge_regions:
            if region[0] != 'conflict':
                yield region
                continue
            (type, iz, zmatch, ia, amatch, ib, bmatch) = region
            a_region = self.a[ia:amatch]
            b_region = self.b[ib:bmatch]
            matches = patiencediff.PatienceSequenceMatcher(None, a_region, b_region).get_matching_blocks()
            next_a = ia
            next_b = ib
            for (region_ia, region_ib, region_len) in matches[:-1]:
                region_ia += ia
                region_ib += ib
                reg = self.mismatch_region(next_a, region_ia, next_b, region_ib)
                if reg is not None:
                    yield reg
                yield ('same', region_ia, region_len + region_ia)
                next_a = region_ia + region_len
                next_b = region_ib + region_len
            reg = self.mismatch_region(next_a, amatch, next_b, bmatch)
            if reg is not None:
                yield reg

    @staticmethod
    def mismatch_region(next_a, region_ia, next_b, region_ib):
        if False:
            for i in range(10):
                print('nop')
        if next_a < region_ia or next_b < region_ib:
            return ('conflict', None, None, next_a, region_ia, next_b, region_ib)

    def find_sync_regions(self):
        if False:
            print('Hello World!')
        'Return a list of sync regions, where both descendents match the base.\n\n        Generates a list of (base1, base2, a1, a2, b1, b2).  There is\n        always a zero-length sync region at the end of all the files.\n        '
        ia = ib = 0
        amatches = patiencediff.PatienceSequenceMatcher(None, self.base, self.a).get_matching_blocks()
        bmatches = patiencediff.PatienceSequenceMatcher(None, self.base, self.b).get_matching_blocks()
        len_a = len(amatches)
        len_b = len(bmatches)
        sl = []
        while ia < len_a and ib < len_b:
            (abase, amatch, alen) = amatches[ia]
            (bbase, bmatch, blen) = bmatches[ib]
            i = intersect((abase, abase + alen), (bbase, bbase + blen))
            if i:
                intbase = i[0]
                intend = i[1]
                intlen = intend - intbase
                asub = amatch + (intbase - abase)
                bsub = bmatch + (intbase - bbase)
                aend = asub + intlen
                bend = bsub + intlen
                sl.append((intbase, intend, asub, aend, bsub, bend))
            if abase + alen < bbase + blen:
                ia += 1
            else:
                ib += 1
        intbase = len(self.base)
        abase = len(self.a)
        bbase = len(self.b)
        sl.append((intbase, intbase, abase, abase, bbase, bbase))
        return sl

    def find_unconflicted(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a list of ranges in base that are not conflicted.'
        am = patiencediff.PatienceSequenceMatcher(None, self.base, self.a).get_matching_blocks()
        bm = patiencediff.PatienceSequenceMatcher(None, self.base, self.b).get_matching_blocks()
        unc = []
        while am and bm:
            a1 = am[0][0]
            a2 = a1 + am[0][2]
            b1 = bm[0][0]
            b2 = b1 + bm[0][2]
            i = intersect((a1, a2), (b1, b2))
            if i:
                unc.append(i)
            if a2 < b2:
                del am[0]
            else:
                del bm[0]
        return unc

def main(argv):
    if False:
        while True:
            i = 10
    a = file(argv[1], 'rt').readlines()
    base = file(argv[2], 'rt').readlines()
    b = file(argv[3], 'rt').readlines()
    m3 = Merge3(base, a, b)
    sys.stdout.writelines(m3.merge_annotated())
if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))