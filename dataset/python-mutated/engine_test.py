import sys
from hscommon.jobprogress import job
from hscommon.util import first
from hscommon.testutil import eq_, log_calls
from core.tests.base import NamedObject
from core import engine
from core.engine import get_match, getwords, Group, getfields, unpack_fields, compare_fields, compare, WEIGHT_WORDS, MATCH_SIMILAR_WORDS, NO_FIELD_ORDER, build_word_dict, get_groups, getmatches, Match, getmatches_by_contents, merge_similar_words, reduce_common_words
no = NamedObject

def get_match_triangle():
    if False:
        return 10
    o1 = NamedObject(with_words=True)
    o2 = NamedObject(with_words=True)
    o3 = NamedObject(with_words=True)
    m1 = get_match(o1, o2)
    m2 = get_match(o1, o3)
    m3 = get_match(o2, o3)
    return [m1, m2, m3]

def get_test_group():
    if False:
        i = 10
        return i + 15
    (m1, m2, m3) = get_match_triangle()
    result = Group()
    result.add_match(m1)
    result.add_match(m2)
    result.add_match(m3)
    return result

def assert_match(m, name1, name2):
    if False:
        return 10
    if m.first.name == name1:
        eq_(m.second.name, name2)
    else:
        eq_(m.first.name, name2)
        eq_(m.second.name, name1)

class TestCasegetwords:

    def test_spaces(self):
        if False:
            print('Hello World!')
        eq_(['a', 'b', 'c', 'd'], getwords('a b c d'))
        eq_(['a', 'b', 'c', 'd'], getwords(' a  b  c d '))

    def test_unicode(self):
        if False:
            return 10
        eq_(['e', 'c', '0', 'a', 'o', 'u', 'e', 'u'], getwords('é ç 0 à ö û è ¤ ù'))
        eq_(['02', '君のこころは輝いてるかい？', '国木田花丸', 'solo', 'ver'], getwords('02 君のこころは輝いてるかい？ 国木田花丸 Solo Ver'))

    def test_splitter_chars(self):
        if False:
            return 10
        eq_([chr(i) for i in range(ord('a'), ord('z') + 1)], getwords('a-b_c&d+e(f)g;h\\i[j]k{l}m:n.o,p<q>r/s?t~u!v@w#x$y*z'))

    def test_joiner_chars(self):
        if False:
            for i in range(10):
                print('nop')
        eq_(['aec'], getwords("a'éc"))

    def test_empty(self):
        if False:
            for i in range(10):
                print('nop')
        eq_([], getwords(''))

    def test_returns_lowercase(self):
        if False:
            print('Hello World!')
        eq_(['foo', 'bar'], getwords('FOO BAR'))

    def test_decompose_unicode(self):
        if False:
            for i in range(10):
                print('nop')
        eq_(['fooebar'], getwords('fooébar'))

class TestCasegetfields:

    def test_simple(self):
        if False:
            print('Hello World!')
        eq_([['a', 'b'], ['c', 'd', 'e']], getfields('a b - c d e'))

    def test_empty(self):
        if False:
            i = 10
            return i + 15
        eq_([], getfields(''))

    def test_cleans_empty_fields(self):
        if False:
            return 10
        expected = [['a', 'bc', 'def']]
        actual = getfields(' - a bc def')
        eq_(expected, actual)

class TestCaseUnpackFields:

    def test_with_fields(self):
        if False:
            print('Hello World!')
        expected = ['a', 'b', 'c', 'd', 'e', 'f']
        actual = unpack_fields([['a'], ['b', 'c'], ['d', 'e', 'f']])
        eq_(expected, actual)

    def test_without_fields(self):
        if False:
            print('Hello World!')
        expected = ['a', 'b', 'c', 'd', 'e', 'f']
        actual = unpack_fields(['a', 'b', 'c', 'd', 'e', 'f'])
        eq_(expected, actual)

    def test_empty(self):
        if False:
            while True:
                i = 10
        eq_([], unpack_fields([]))

class TestCaseWordCompare:

    def test_list(self):
        if False:
            print('Hello World!')
        eq_(100, compare(['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd']))
        eq_(86, compare(['a', 'b', 'c', 'd'], ['a', 'b', 'c']))

    def test_unordered(self):
        if False:
            i = 10
            return i + 15
        eq_(99, compare(['a', 'b', 'c', 'd'], ['d', 'b', 'c', 'a']))

    def test_word_occurs_twice(self):
        if False:
            i = 10
            return i + 15
        eq_(89, compare(['a', 'b', 'c', 'd', 'a'], ['d', 'b', 'c', 'a']))

    def test_uses_copy_of_lists(self):
        if False:
            for i in range(10):
                print('nop')
        first = ['foo', 'bar']
        second = ['bar', 'bleh']
        compare(first, second)
        eq_(['foo', 'bar'], first)
        eq_(['bar', 'bleh'], second)

    def test_word_weight(self):
        if False:
            while True:
                i = 10
        eq_(int(6.0 / 13.0 * 100), compare(['foo', 'bar'], ['bar', 'bleh'], (WEIGHT_WORDS,)))

    def test_similar_words(self):
        if False:
            i = 10
            return i + 15
        eq_(100, compare(['the', 'white', 'stripes'], ['the', 'whites', 'stripe'], (MATCH_SIMILAR_WORDS,)))

    def test_empty(self):
        if False:
            return 10
        eq_(0, compare([], []))

    def test_with_fields(self):
        if False:
            while True:
                i = 10
        eq_(67, compare([['a', 'b'], ['c', 'd', 'e']], [['a', 'b'], ['c', 'd', 'f']]))

    def test_propagate_flags_with_fields(self, monkeypatch):
        if False:
            for i in range(10):
                print('nop')

        def mock_compare(first, second, flags):
            if False:
                while True:
                    i = 10
            eq_((0, 1, 2, 3, 5), flags)
        monkeypatch.setattr(engine, 'compare_fields', mock_compare)
        compare([['a']], [['a']], (0, 1, 2, 3, 5))

class TestCaseWordCompareWithFields:

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        eq_(67, compare_fields([['a', 'b'], ['c', 'd', 'e']], [['a', 'b'], ['c', 'd', 'f']]))

    def test_empty(self):
        if False:
            print('Hello World!')
        eq_(0, compare_fields([], []))

    def test_different_length(self):
        if False:
            while True:
                i = 10
        eq_(0, compare_fields([['a'], ['b']], [['a'], ['b'], ['c']]))

    def test_propagates_flags(self, monkeypatch):
        if False:
            for i in range(10):
                print('nop')

        def mock_compare(first, second, flags):
            if False:
                i = 10
                return i + 15
            eq_((0, 1, 2, 3, 5), flags)
        monkeypatch.setattr(engine, 'compare_fields', mock_compare)
        compare_fields([['a']], [['a']], (0, 1, 2, 3, 5))

    def test_order(self):
        if False:
            while True:
                i = 10
        first = [['a', 'b'], ['c', 'd', 'e']]
        second = [['c', 'd', 'f'], ['a', 'b']]
        eq_(0, compare_fields(first, second))

    def test_no_order(self):
        if False:
            return 10
        first = [['a', 'b'], ['c', 'd', 'e']]
        second = [['c', 'd', 'f'], ['a', 'b']]
        eq_(67, compare_fields(first, second, (NO_FIELD_ORDER,)))
        first = [['a', 'b'], ['a', 'b']]
        second = [['c', 'd', 'f'], ['a', 'b']]
        eq_(0, compare_fields(first, second, (NO_FIELD_ORDER,)))
        first = [['a', 'b'], ['a', 'b', 'c']]
        second = [['c', 'd', 'f'], ['a', 'b']]
        eq_(33, compare_fields(first, second, (NO_FIELD_ORDER,)))

    def test_compare_fields_without_order_doesnt_alter_fields(self):
        if False:
            i = 10
            return i + 15
        first = [['a', 'b'], ['c', 'd', 'e']]
        second = [['c', 'd', 'f'], ['a', 'b']]
        eq_(67, compare_fields(first, second, (NO_FIELD_ORDER,)))
        eq_([['a', 'b'], ['c', 'd', 'e']], first)
        eq_([['c', 'd', 'f'], ['a', 'b']], second)

class TestCaseBuildWordDict:

    def test_with_standard_words(self):
        if False:
            i = 10
            return i + 15
        item_list = [NamedObject('foo bar', True)]
        item_list.append(NamedObject('bar baz', True))
        item_list.append(NamedObject('baz bleh foo', True))
        d = build_word_dict(item_list)
        eq_(4, len(d))
        eq_(2, len(d['foo']))
        assert item_list[0] in d['foo']
        assert item_list[2] in d['foo']
        eq_(2, len(d['bar']))
        assert item_list[0] in d['bar']
        assert item_list[1] in d['bar']
        eq_(2, len(d['baz']))
        assert item_list[1] in d['baz']
        assert item_list[2] in d['baz']
        eq_(1, len(d['bleh']))
        assert item_list[2] in d['bleh']

    def test_unpack_fields(self):
        if False:
            for i in range(10):
                print('nop')
        o = NamedObject('')
        o.words = [['foo', 'bar'], ['baz']]
        d = build_word_dict([o])
        eq_(3, len(d))
        eq_(1, len(d['foo']))

    def test_words_are_unaltered(self):
        if False:
            i = 10
            return i + 15
        o = NamedObject('')
        o.words = [['foo', 'bar'], ['baz']]
        build_word_dict([o])
        eq_([['foo', 'bar'], ['baz']], o.words)

    def test_object_instances_can_only_be_once_in_words_object_list(self):
        if False:
            for i in range(10):
                print('nop')
        o = NamedObject('foo foo', True)
        d = build_word_dict([o])
        eq_(1, len(d['foo']))

    def test_job(self):
        if False:
            i = 10
            return i + 15

        def do_progress(p, d=''):
            if False:
                for i in range(10):
                    print('nop')
            self.log.append(p)
            return True
        j = job.Job(1, do_progress)
        self.log = []
        s = 'foo bar'
        build_word_dict([NamedObject(s, True), NamedObject(s, True), NamedObject(s, True)], j)
        eq_(0, self.log[0])
        eq_(100, self.log[1])

class TestCaseMergeSimilarWords:

    def test_some_similar_words(self):
        if False:
            return 10
        d = {'foobar': {1}, 'foobar1': {2}, 'foobar2': {3}}
        merge_similar_words(d)
        eq_(1, len(d))
        eq_(3, len(d['foobar']))

class TestCaseReduceCommonWords:

    def test_typical(self):
        if False:
            i = 10
            return i + 15
        d = {'foo': {NamedObject('foo bar', True) for _ in range(50)}, 'bar': {NamedObject('foo bar', True) for _ in range(49)}}
        reduce_common_words(d, 50)
        assert 'foo' not in d
        eq_(49, len(d['bar']))

    def test_dont_remove_objects_with_only_common_words(self):
        if False:
            while True:
                i = 10
        d = {'common': set([NamedObject('common uncommon', True) for _ in range(50)] + [NamedObject('common', True)]), 'uncommon': {NamedObject('common uncommon', True)}}
        reduce_common_words(d, 50)
        eq_(1, len(d['common']))
        eq_(1, len(d['uncommon']))

    def test_values_still_are_set_instances(self):
        if False:
            return 10
        d = {'common': set([NamedObject('common uncommon', True) for _ in range(50)] + [NamedObject('common', True)]), 'uncommon': {NamedObject('common uncommon', True)}}
        reduce_common_words(d, 50)
        assert isinstance(d['common'], set)
        assert isinstance(d['uncommon'], set)

    def test_dont_raise_keyerror_when_a_word_has_been_removed(self):
        if False:
            return 10
        d = {'foo': {NamedObject('foo bar baz', True) for _ in range(50)}, 'bar': {NamedObject('foo bar baz', True) for _ in range(50)}, 'baz': {NamedObject('foo bar baz', True) for _ in range(49)}}
        try:
            reduce_common_words(d, 50)
        except KeyError:
            self.fail()

    def test_unpack_fields(self):
        if False:
            print('Hello World!')

        def create_it():
            if False:
                print('Hello World!')
            o = NamedObject('')
            o.words = [['foo', 'bar'], ['baz']]
            return o
        d = {'foo': {create_it() for _ in range(50)}}
        try:
            reduce_common_words(d, 50)
        except TypeError:
            self.fail('must support fields.')

    def test_consider_a_reduced_common_word_common_even_after_reduction(self):
        if False:
            i = 10
            return i + 15
        only_common = NamedObject('foo bar', True)
        d = {'foo': set([NamedObject('foo bar baz', True) for _ in range(49)] + [only_common]), 'bar': set([NamedObject('foo bar baz', True) for _ in range(49)] + [only_common]), 'baz': {NamedObject('foo bar baz', True) for _ in range(49)}}
        reduce_common_words(d, 50)
        eq_(1, len(d['foo']))
        eq_(1, len(d['bar']))
        eq_(49, len(d['baz']))

class TestCaseGetMatch:

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        o1 = NamedObject('foo bar', True)
        o2 = NamedObject('bar bleh', True)
        m = get_match(o1, o2)
        eq_(50, m.percentage)
        eq_(['foo', 'bar'], m.first.words)
        eq_(['bar', 'bleh'], m.second.words)
        assert m.first is o1
        assert m.second is o2

    def test_in(self):
        if False:
            while True:
                i = 10
        o1 = NamedObject('foo', True)
        o2 = NamedObject('bar', True)
        m = get_match(o1, o2)
        assert o1 in m
        assert o2 in m
        assert object() not in m

    def test_word_weight(self):
        if False:
            while True:
                i = 10
        m = get_match(NamedObject('foo bar', True), NamedObject('bar bleh', True), (WEIGHT_WORDS,))
        eq_(m.percentage, int(6.0 / 13.0 * 100))

class TestCaseGetMatches:

    def test_empty(self):
        if False:
            print('Hello World!')
        eq_(getmatches([]), [])

    def test_simple(self):
        if False:
            print('Hello World!')
        item_list = [NamedObject('foo bar'), NamedObject('bar bleh'), NamedObject('a b c foo')]
        r = getmatches(item_list)
        eq_(2, len(r))
        m = first((m for m in r if m.percentage == 50))
        assert_match(m, 'foo bar', 'bar bleh')
        m = first((m for m in r if m.percentage == 33))
        assert_match(m, 'foo bar', 'a b c foo')

    def test_null_and_unrelated_objects(self):
        if False:
            for i in range(10):
                print('nop')
        item_list = [NamedObject('foo bar'), NamedObject('bar bleh'), NamedObject(''), NamedObject('unrelated object')]
        r = getmatches(item_list)
        eq_(len(r), 1)
        m = r[0]
        eq_(m.percentage, 50)
        assert_match(m, 'foo bar', 'bar bleh')

    def test_twice_the_same_word(self):
        if False:
            while True:
                i = 10
        item_list = [NamedObject('foo foo bar'), NamedObject('bar bleh')]
        r = getmatches(item_list)
        eq_(1, len(r))

    def test_twice_the_same_word_when_preworded(self):
        if False:
            i = 10
            return i + 15
        item_list = [NamedObject('foo foo bar', True), NamedObject('bar bleh', True)]
        r = getmatches(item_list)
        eq_(1, len(r))

    def test_two_words_match(self):
        if False:
            i = 10
            return i + 15
        item_list = [NamedObject('foo bar'), NamedObject('foo bar bleh')]
        r = getmatches(item_list)
        eq_(1, len(r))

    def test_match_files_with_only_common_words(self):
        if False:
            return 10
        item_list = [NamedObject('foo') for _ in range(50)]
        r = getmatches(item_list)
        eq_(1225, len(r))

    def test_use_words_already_there_if_there(self):
        if False:
            i = 10
            return i + 15
        o1 = NamedObject('foo')
        o2 = NamedObject('bar')
        o2.words = ['foo']
        eq_(1, len(getmatches([o1, o2])))

    def test_job(self):
        if False:
            return 10

        def do_progress(p, d=''):
            if False:
                return 10
            self.log.append(p)
            return True
        j = job.Job(1, do_progress)
        self.log = []
        s = 'foo bar'
        getmatches([NamedObject(s), NamedObject(s), NamedObject(s)], j=j)
        assert len(self.log) > 2
        eq_(0, self.log[0])
        eq_(100, self.log[-1])

    def test_weight_words(self):
        if False:
            for i in range(10):
                print('nop')
        item_list = [NamedObject('foo bar'), NamedObject('bar bleh')]
        m = getmatches(item_list, weight_words=True)[0]
        eq_(int(6.0 / 13.0 * 100), m.percentage)

    def test_similar_word(self):
        if False:
            while True:
                i = 10
        item_list = [NamedObject('foobar'), NamedObject('foobars')]
        eq_(len(getmatches(item_list, match_similar_words=True)), 1)
        eq_(getmatches(item_list, match_similar_words=True)[0].percentage, 100)
        item_list = [NamedObject('foobar'), NamedObject('foo')]
        eq_(len(getmatches(item_list, match_similar_words=True)), 0)
        item_list = [NamedObject('bizkit'), NamedObject('bizket')]
        eq_(len(getmatches(item_list, match_similar_words=True)), 1)
        item_list = [NamedObject('foobar'), NamedObject('foosbar')]
        eq_(len(getmatches(item_list, match_similar_words=True)), 1)

    def test_single_object_with_similar_words(self):
        if False:
            i = 10
            return i + 15
        item_list = [NamedObject('foo foos')]
        eq_(len(getmatches(item_list, match_similar_words=True)), 0)

    def test_double_words_get_counted_only_once(self):
        if False:
            i = 10
            return i + 15
        item_list = [NamedObject('foo bar foo bleh'), NamedObject('foo bar bleh bar')]
        m = getmatches(item_list)[0]
        eq_(75, m.percentage)

    def test_with_fields(self):
        if False:
            return 10
        o1 = NamedObject('foo bar - foo bleh')
        o2 = NamedObject('foo bar - bleh bar')
        o1.words = getfields(o1.name)
        o2.words = getfields(o2.name)
        m = getmatches([o1, o2])[0]
        eq_(50, m.percentage)

    def test_with_fields_no_order(self):
        if False:
            while True:
                i = 10
        o1 = NamedObject('foo bar - foo bleh')
        o2 = NamedObject('bleh bang - foo bar')
        o1.words = getfields(o1.name)
        o2.words = getfields(o2.name)
        m = getmatches([o1, o2], no_field_order=True)[0]
        eq_(m.percentage, 50)

    def test_only_match_similar_when_the_option_is_set(self):
        if False:
            for i in range(10):
                print('nop')
        item_list = [NamedObject('foobar'), NamedObject('foobars')]
        eq_(len(getmatches(item_list, match_similar_words=False)), 0)

    def test_dont_recurse_do_match(self):
        if False:
            i = 10
            return i + 15
        sys.setrecursionlimit(200)
        files = [NamedObject('foo bar') for _ in range(201)]
        try:
            getmatches(files)
        except RuntimeError:
            self.fail()
        finally:
            sys.setrecursionlimit(1000)

    def test_min_match_percentage(self):
        if False:
            i = 10
            return i + 15
        item_list = [NamedObject('foo bar'), NamedObject('bar bleh'), NamedObject('a b c foo')]
        r = getmatches(item_list, min_match_percentage=50)
        eq_(1, len(r))

    def test_memory_error(self, monkeypatch):
        if False:
            return 10

        @log_calls
        def mocked_match(first, second, flags):
            if False:
                while True:
                    i = 10
            if len(mocked_match.calls) > 42:
                raise MemoryError()
            return Match(first, second, 0)
        objects = [NamedObject() for _ in range(10)]
        monkeypatch.setattr(engine, 'get_match', mocked_match)
        try:
            r = getmatches(objects)
        except MemoryError:
            self.fail('MemoryError must be handled')
        eq_(42, len(r))

class TestCaseGetMatchesByContents:

    def test_big_file_partial_hashing(self):
        if False:
            return 10
        smallsize = 1
        bigsize = 100 * 1024 * 1024
        f = [no('bigfoo', size=bigsize), no('bigbar', size=bigsize), no('smallfoo', size=smallsize), no('smallbar', size=smallsize)]
        f[0].digest = f[0].digest_partial = f[0].digest_samples = 'foobar'
        f[1].digest = f[1].digest_partial = f[1].digest_samples = 'foobar'
        f[2].digest = f[2].digest_partial = 'bleh'
        f[3].digest = f[3].digest_partial = 'bleh'
        r = getmatches_by_contents(f, bigsize=bigsize)
        eq_(len(r), 2)
        r = getmatches_by_contents(f, bigsize=0)
        eq_(len(r), 2)
        f[1].digest = f[1].digest_samples = 'foobardiff'
        r = getmatches_by_contents(f, bigsize=bigsize)
        eq_(len(r), 1)
        r = getmatches_by_contents(f, bigsize=0)
        eq_(len(r), 1)

class TestCaseGroup:

    def test_empty(self):
        if False:
            for i in range(10):
                print('nop')
        g = Group()
        eq_(None, g.ref)
        eq_([], g.dupes)
        eq_(0, len(g.matches))

    def test_add_match(self):
        if False:
            for i in range(10):
                print('nop')
        g = Group()
        m = get_match(NamedObject('foo', True), NamedObject('bar', True))
        g.add_match(m)
        assert g.ref is m.first
        eq_([m.second], g.dupes)
        eq_(1, len(g.matches))
        assert m in g.matches

    def test_multiple_add_match(self):
        if False:
            while True:
                i = 10
        g = Group()
        o1 = NamedObject('a', True)
        o2 = NamedObject('b', True)
        o3 = NamedObject('c', True)
        o4 = NamedObject('d', True)
        g.add_match(get_match(o1, o2))
        assert g.ref is o1
        eq_([o2], g.dupes)
        eq_(1, len(g.matches))
        g.add_match(get_match(o1, o3))
        eq_([o2], g.dupes)
        eq_(2, len(g.matches))
        g.add_match(get_match(o2, o3))
        eq_([o2, o3], g.dupes)
        eq_(3, len(g.matches))
        g.add_match(get_match(o1, o4))
        eq_([o2, o3], g.dupes)
        eq_(4, len(g.matches))
        g.add_match(get_match(o2, o4))
        eq_([o2, o3], g.dupes)
        eq_(5, len(g.matches))
        g.add_match(get_match(o3, o4))
        eq_([o2, o3, o4], g.dupes)
        eq_(6, len(g.matches))

    def test_len(self):
        if False:
            print('Hello World!')
        g = Group()
        eq_(0, len(g))
        g.add_match(get_match(NamedObject('foo', True), NamedObject('bar', True)))
        eq_(2, len(g))

    def test_add_same_match_twice(self):
        if False:
            for i in range(10):
                print('nop')
        g = Group()
        m = get_match(NamedObject('foo', True), NamedObject('foo', True))
        g.add_match(m)
        eq_(2, len(g))
        eq_(1, len(g.matches))
        g.add_match(m)
        eq_(2, len(g))
        eq_(1, len(g.matches))

    def test_in(self):
        if False:
            while True:
                i = 10
        g = Group()
        o1 = NamedObject('foo', True)
        o2 = NamedObject('bar', True)
        assert o1 not in g
        g.add_match(get_match(o1, o2))
        assert o1 in g
        assert o2 in g

    def test_remove(self):
        if False:
            return 10
        g = Group()
        o1 = NamedObject('foo', True)
        o2 = NamedObject('bar', True)
        o3 = NamedObject('bleh', True)
        g.add_match(get_match(o1, o2))
        g.add_match(get_match(o1, o3))
        g.add_match(get_match(o2, o3))
        eq_(3, len(g.matches))
        eq_(3, len(g))
        g.remove_dupe(o3)
        eq_(1, len(g.matches))
        eq_(2, len(g))
        g.remove_dupe(o1)
        eq_(0, len(g.matches))
        eq_(0, len(g))

    def test_remove_with_ref_dupes(self):
        if False:
            print('Hello World!')
        g = Group()
        o1 = NamedObject('foo', True)
        o2 = NamedObject('bar', True)
        o3 = NamedObject('bleh', True)
        g.add_match(get_match(o1, o2))
        g.add_match(get_match(o1, o3))
        g.add_match(get_match(o2, o3))
        o1.is_ref = True
        o2.is_ref = True
        g.remove_dupe(o3)
        eq_(0, len(g))

    def test_switch_ref(self):
        if False:
            for i in range(10):
                print('nop')
        o1 = NamedObject(with_words=True)
        o2 = NamedObject(with_words=True)
        g = Group()
        g.add_match(get_match(o1, o2))
        assert o1 is g.ref
        g.switch_ref(o2)
        assert o2 is g.ref
        eq_([o1], g.dupes)
        g.switch_ref(o2)
        assert o2 is g.ref
        g.switch_ref(NamedObject('', True))
        assert o2 is g.ref

    def test_switch_ref_from_ref_dir(self):
        if False:
            return 10
        o1 = no(with_words=True)
        o2 = no(with_words=True)
        o1.is_ref = True
        g = Group()
        g.add_match(get_match(o1, o2))
        g.switch_ref(o2)
        assert o1 is g.ref

    def test_get_match_of(self):
        if False:
            while True:
                i = 10
        g = Group()
        for m in get_match_triangle():
            g.add_match(m)
        o = g.dupes[0]
        m = g.get_match_of(o)
        assert g.ref in m
        assert o in m
        assert g.get_match_of(NamedObject('', True)) is None
        assert g.get_match_of(g.ref) is None

    def test_percentage(self):
        if False:
            i = 10
            return i + 15
        (m1, m2, m3) = get_match_triangle()
        m1 = Match(m1[0], m1[1], 100)
        m2 = Match(m2[0], m2[1], 50)
        m3 = Match(m3[0], m3[1], 33)
        g = Group()
        g.add_match(m1)
        g.add_match(m2)
        g.add_match(m3)
        eq_(75, g.percentage)
        g.switch_ref(g.dupes[0])
        eq_(66, g.percentage)
        g.remove_dupe(g.dupes[0])
        eq_(33, g.percentage)
        g.add_match(m1)
        g.add_match(m2)
        eq_(66, g.percentage)

    def test_percentage_on_empty_group(self):
        if False:
            while True:
                i = 10
        g = Group()
        eq_(0, g.percentage)

    def test_prioritize(self):
        if False:
            for i in range(10):
                print('nop')
        (m1, m2, m3) = get_match_triangle()
        o1 = m1.first
        o2 = m1.second
        o3 = m2.second
        o1.name = 'c'
        o2.name = 'b'
        o3.name = 'a'
        g = Group()
        g.add_match(m1)
        g.add_match(m2)
        g.add_match(m3)
        assert o1 is g.ref
        assert g.prioritize(lambda x: x.name)
        assert o3 is g.ref

    def test_prioritize_with_tie_breaker(self):
        if False:
            for i in range(10):
                print('nop')
        g = get_test_group()
        (o1, o2, o3) = g.ordered
        g.prioritize(lambda x: 0, lambda ref, dupe: dupe is o3)
        assert g.ref is o3

    def test_prioritize_with_tie_breaker_runs_on_all_dupes(self):
        if False:
            for i in range(10):
                print('nop')
        g = get_test_group()
        (o1, o2, o3) = g.ordered
        o1.foo = 1
        o2.foo = 2
        o3.foo = 3
        g.prioritize(lambda x: 0, lambda ref, dupe: dupe.foo > ref.foo)
        assert g.ref is o3

    def test_prioritize_with_tie_breaker_runs_only_on_tie_dupes(self):
        if False:
            while True:
                i = 10
        g = get_test_group()
        (o1, o2, o3) = g.ordered
        o1.foo = 2
        o2.foo = 2
        o3.foo = 1
        o1.bar = 1
        o2.bar = 2
        o3.bar = 3
        g.prioritize(lambda x: -x.foo, lambda ref, dupe: dupe.bar > ref.bar)
        assert g.ref is o2

    def test_prioritize_with_ref_dupe(self):
        if False:
            i = 10
            return i + 15
        g = get_test_group()
        (o1, o2, o3) = g
        o1.is_ref = True
        o2.size = 2
        g.prioritize(lambda x: -x.size)
        assert g.ref is o1

    def test_prioritize_nothing_changes(self):
        if False:
            while True:
                i = 10
        g = get_test_group()
        g[0].name = 'a'
        g[1].name = 'b'
        g[2].name = 'c'
        assert not g.prioritize(lambda x: x.name)

    def test_list_like(self):
        if False:
            return 10
        g = Group()
        (o1, o2) = (NamedObject('foo', True), NamedObject('bar', True))
        g.add_match(get_match(o1, o2))
        assert g[0] is o1
        assert g[1] is o2

    def test_discard_matches(self):
        if False:
            while True:
                i = 10
        g = Group()
        (o1, o2, o3) = (NamedObject('foo', True), NamedObject('bar', True), NamedObject('baz', True))
        g.add_match(get_match(o1, o2))
        g.add_match(get_match(o1, o3))
        g.discard_matches()
        eq_(1, len(g.matches))
        eq_(0, len(g.candidates))

class TestCaseGetGroups:

    def test_empty(self):
        if False:
            for i in range(10):
                print('nop')
        r = get_groups([])
        eq_([], r)

    def test_simple(self):
        if False:
            return 10
        item_list = [NamedObject('foo bar'), NamedObject('bar bleh')]
        matches = getmatches(item_list)
        m = matches[0]
        r = get_groups(matches)
        eq_(1, len(r))
        g = r[0]
        assert g.ref is m.first
        eq_([m.second], g.dupes)

    def test_group_with_multiple_matches(self):
        if False:
            while True:
                i = 10
        item_list = [NamedObject('foo'), NamedObject('foo'), NamedObject('foo')]
        matches = getmatches(item_list)
        r = get_groups(matches)
        eq_(1, len(r))
        g = r[0]
        eq_(3, len(g))

    def test_must_choose_a_group(self):
        if False:
            return 10
        item_list = [NamedObject('a b'), NamedObject('a b'), NamedObject('b c'), NamedObject('c d'), NamedObject('c d')]
        matches = getmatches(item_list)
        r = get_groups(matches)
        eq_(2, len(r))
        eq_(5, len(r[0]) + len(r[1]))

    def test_should_all_go_in_the_same_group(self):
        if False:
            for i in range(10):
                print('nop')
        item_list = [NamedObject('a b'), NamedObject('a b'), NamedObject('a b'), NamedObject('a b')]
        matches = getmatches(item_list)
        r = get_groups(matches)
        eq_(1, len(r))

    def test_give_priority_to_matches_with_higher_percentage(self):
        if False:
            i = 10
            return i + 15
        o1 = NamedObject(with_words=True)
        o2 = NamedObject(with_words=True)
        o3 = NamedObject(with_words=True)
        m1 = Match(o1, o2, 1)
        m2 = Match(o2, o3, 2)
        r = get_groups([m1, m2])
        eq_(1, len(r))
        g = r[0]
        eq_(2, len(g))
        assert o1 not in g
        assert o2 in g
        assert o3 in g

    def test_four_sized_group(self):
        if False:
            while True:
                i = 10
        item_list = [NamedObject('foobar') for _ in range(4)]
        m = getmatches(item_list)
        r = get_groups(m)
        eq_(1, len(r))
        eq_(4, len(r[0]))

    def test_referenced_by_ref2(self):
        if False:
            while True:
                i = 10
        o1 = NamedObject(with_words=True)
        o2 = NamedObject(with_words=True)
        o3 = NamedObject(with_words=True)
        m1 = get_match(o1, o2)
        m2 = get_match(o3, o1)
        m3 = get_match(o3, o2)
        r = get_groups([m1, m2, m3])
        eq_(3, len(r[0]))

    def test_group_admissible_discarded_dupes(self):
        if False:
            print('Hello World!')
        (A, B, C, D) = (NamedObject() for _ in range(4))
        m1 = Match(A, B, 90)
        m2 = Match(A, C, 80)
        m3 = Match(A, D, 80)
        m4 = Match(C, D, 70)
        groups = get_groups([m1, m2, m3, m4])
        eq_(len(groups), 2)
        (g1, g2) = groups
        assert A in g1
        assert B in g1
        assert C in g2
        assert D in g2