"""Tests for the python and pyrex extensions of groupcompress"""
from bzrlib import _groupcompress_py, tests
from bzrlib.tests.scenarios import load_tests_apply_scenarios
from bzrlib.tests import features

def module_scenarios():
    if False:
        i = 10
        return i + 15
    scenarios = [('python', {'_gc_module': _groupcompress_py})]
    if compiled_groupcompress_feature.available():
        gc_module = compiled_groupcompress_feature.module
        scenarios.append(('C', {'_gc_module': gc_module}))
    return scenarios

def two_way_scenarios():
    if False:
        for i in range(10):
            print('nop')
    scenarios = [('PP', {'make_delta': _groupcompress_py.make_delta, 'apply_delta': _groupcompress_py.apply_delta})]
    if compiled_groupcompress_feature.available():
        gc_module = compiled_groupcompress_feature.module
        scenarios.extend([('CC', {'make_delta': gc_module.make_delta, 'apply_delta': gc_module.apply_delta}), ('PC', {'make_delta': _groupcompress_py.make_delta, 'apply_delta': gc_module.apply_delta}), ('CP', {'make_delta': gc_module.make_delta, 'apply_delta': _groupcompress_py.apply_delta})])
    return scenarios
load_tests = load_tests_apply_scenarios
compiled_groupcompress_feature = features.ModuleAvailableFeature('bzrlib._groupcompress_pyx')
_text1 = 'This is a bit\nof source text\nwhich is meant to be matched\nagainst other text\n'
_text2 = 'This is a bit\nof source text\nwhich is meant to differ from\nagainst other text\n'
_text3 = 'This is a bit\nof source text\nwhich is meant to be matched\nagainst other text\nexcept it also\nhas a lot more data\nat the end of the file\n'
_first_text = 'a bit of text, that\ndoes not have much in\ncommon with the next text\n'
_second_text = 'some more bit of text, that\ndoes not have much in\ncommon with the previous text\nand has some extra text\n'
_third_text = 'a bit of text, that\nhas some in common with the previous text\nand has some extra text\nand not have much in\ncommon with the next text\n'
_fourth_text = '123456789012345\nsame rabin hash\n123456789012345\nsame rabin hash\n123456789012345\nsame rabin hash\n123456789012345\nsame rabin hash\n'

class TestMakeAndApplyDelta(tests.TestCase):
    scenarios = module_scenarios()
    _gc_module = None

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestMakeAndApplyDelta, self).setUp()
        self.make_delta = self._gc_module.make_delta
        self.apply_delta = self._gc_module.apply_delta
        self.apply_delta_to_source = self._gc_module.apply_delta_to_source

    def test_make_delta_is_typesafe(self):
        if False:
            print('Hello World!')
        self.make_delta('a string', 'another string')

        def _check_make_delta(string1, string2):
            if False:
                return 10
            self.assertRaises(TypeError, self.make_delta, string1, string2)
        _check_make_delta('a string', object())
        _check_make_delta('a string', u'not a string')
        _check_make_delta(object(), 'a string')
        _check_make_delta(u'not a string', 'a string')

    def test_make_noop_delta(self):
        if False:
            return 10
        ident_delta = self.make_delta(_text1, _text1)
        self.assertEqual('M\x90M', ident_delta)
        ident_delta = self.make_delta(_text2, _text2)
        self.assertEqual('N\x90N', ident_delta)
        ident_delta = self.make_delta(_text3, _text3)
        self.assertEqual('\x87\x01\x90\x87', ident_delta)

    def assertDeltaIn(self, delta1, delta2, delta):
        if False:
            while True:
                i = 10
        'Make sure that the delta bytes match one of the expectations.'
        if delta not in (delta1, delta2):
            self.fail('Delta bytes:\n       %r\nnot in %r\n    or %r' % (delta, delta1, delta2))

    def test_make_delta(self):
        if False:
            for i in range(10):
                print('nop')
        delta = self.make_delta(_text1, _text2)
        self.assertDeltaIn('N\x90/\x1fdiffer from\nagainst other text\n', 'N\x90\x1d\x1ewhich is meant to differ from\n\x91:\x13', delta)
        delta = self.make_delta(_text2, _text1)
        self.assertDeltaIn('M\x90/\x1ebe matched\nagainst other text\n', 'M\x90\x1d\x1dwhich is meant to be matched\n\x91;\x13', delta)
        delta = self.make_delta(_text3, _text1)
        self.assertEqual('M\x90M', delta)
        delta = self.make_delta(_text3, _text2)
        self.assertDeltaIn('N\x90/\x1fdiffer from\nagainst other text\n', 'N\x90\x1d\x1ewhich is meant to differ from\n\x91:\x13', delta)

    def test_make_delta_with_large_copies(self):
        if False:
            i = 10
            return i + 15
        big_text = _text3 * 1220
        delta = self.make_delta(big_text, big_text)
        self.assertDeltaIn('Ü\x86\n\x80\x84\x01´\x02\\\x83', None, delta)

    def test_apply_delta_is_typesafe(self):
        if False:
            for i in range(10):
                print('nop')
        self.apply_delta(_text1, 'M\x90M')
        self.assertRaises(TypeError, self.apply_delta, object(), 'M\x90M')
        self.assertRaises(TypeError, self.apply_delta, unicode(_text1), 'M\x90M')
        self.assertRaises(TypeError, self.apply_delta, _text1, u'M\x90M')
        self.assertRaises(TypeError, self.apply_delta, _text1, object())

    def test_apply_delta(self):
        if False:
            print('Hello World!')
        target = self.apply_delta(_text1, 'N\x90/\x1fdiffer from\nagainst other text\n')
        self.assertEqual(_text2, target)
        target = self.apply_delta(_text2, 'M\x90/\x1ebe matched\nagainst other text\n')
        self.assertEqual(_text1, target)

    def test_apply_delta_to_source_is_safe(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, self.apply_delta_to_source, object(), 0, 1)
        self.assertRaises(TypeError, self.apply_delta_to_source, u'unicode str', 0, 1)
        self.assertRaises(ValueError, self.apply_delta_to_source, 'foo', 1, 4)
        self.assertRaises(ValueError, self.apply_delta_to_source, 'foo', 5, 3)
        self.assertRaises(ValueError, self.apply_delta_to_source, 'foo', 3, 2)

    def test_apply_delta_to_source(self):
        if False:
            return 10
        source_and_delta = _text1 + 'N\x90/\x1fdiffer from\nagainst other text\n'
        self.assertEqual(_text2, self.apply_delta_to_source(source_and_delta, len(_text1), len(source_and_delta)))

class TestMakeAndApplyCompatible(tests.TestCase):
    scenarios = two_way_scenarios()
    make_delta = None
    apply_delta = None

    def assertMakeAndApply(self, source, target):
        if False:
            return 10
        'Assert that generating a delta and applying gives success.'
        delta = self.make_delta(source, target)
        bytes = self.apply_delta(source, delta)
        self.assertEqualDiff(target, bytes)

    def test_direct(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertMakeAndApply(_text1, _text2)
        self.assertMakeAndApply(_text2, _text1)
        self.assertMakeAndApply(_text1, _text3)
        self.assertMakeAndApply(_text3, _text1)
        self.assertMakeAndApply(_text2, _text3)
        self.assertMakeAndApply(_text3, _text2)

class TestDeltaIndex(tests.TestCase):

    def setUp(self):
        if False:
            return 10
        super(TestDeltaIndex, self).setUp()
        self.requireFeature(compiled_groupcompress_feature)
        self._gc_module = compiled_groupcompress_feature.module

    def test_repr(self):
        if False:
            return 10
        di = self._gc_module.DeltaIndex('test text\n')
        self.assertEqual('DeltaIndex(1, 10)', repr(di))

    def test__dump_no_index(self):
        if False:
            return 10
        di = self._gc_module.DeltaIndex()
        self.assertEqual(None, di._dump_index())

    def test__dump_index_simple(self):
        if False:
            print('Hello World!')
        di = self._gc_module.DeltaIndex()
        di.add_source(_text1, 0)
        self.assertFalse(di._has_index())
        self.assertEqual(None, di._dump_index())
        _ = di.make_delta(_text1)
        self.assertTrue(di._has_index())
        (hash_list, entry_list) = di._dump_index()
        self.assertEqual(16, len(hash_list))
        self.assertEqual(68, len(entry_list))
        just_entries = [(idx, text_offset, hash_val) for (idx, (text_offset, hash_val)) in enumerate(entry_list) if text_offset != 0 or hash_val != 0]
        rabin_hash = self._gc_module._rabin_hash
        self.assertEqual([(8, 16, rabin_hash(_text1[1:17])), (25, 48, rabin_hash(_text1[33:49])), (34, 32, rabin_hash(_text1[17:33])), (47, 64, rabin_hash(_text1[49:65]))], just_entries)
        for (entry_idx, text_offset, hash_val) in just_entries:
            self.assertEqual(entry_idx, hash_list[hash_val & 15])

    def test__dump_index_two_sources(self):
        if False:
            print('Hello World!')
        di = self._gc_module.DeltaIndex()
        di.add_source(_text1, 0)
        di.add_source(_text2, 2)
        start2 = len(_text1) + 2
        self.assertTrue(di._has_index())
        (hash_list, entry_list) = di._dump_index()
        self.assertEqual(16, len(hash_list))
        self.assertEqual(68, len(entry_list))
        just_entries = [(idx, text_offset, hash_val) for (idx, (text_offset, hash_val)) in enumerate(entry_list) if text_offset != 0 or hash_val != 0]
        rabin_hash = self._gc_module._rabin_hash
        self.assertEqual([(8, 16, rabin_hash(_text1[1:17])), (9, start2 + 16, rabin_hash(_text2[1:17])), (25, 48, rabin_hash(_text1[33:49])), (30, start2 + 64, rabin_hash(_text2[49:65])), (34, 32, rabin_hash(_text1[17:33])), (35, start2 + 32, rabin_hash(_text2[17:33])), (43, start2 + 48, rabin_hash(_text2[33:49])), (47, 64, rabin_hash(_text1[49:65]))], just_entries)
        for (entry_idx, text_offset, hash_val) in just_entries:
            hash_idx = hash_val & 15
            self.assertTrue(hash_list[hash_idx] <= entry_idx < hash_list[hash_idx + 1])

    def test_first_add_source_doesnt_index_until_make_delta(self):
        if False:
            print('Hello World!')
        di = self._gc_module.DeltaIndex()
        self.assertFalse(di._has_index())
        di.add_source(_text1, 0)
        self.assertFalse(di._has_index())
        delta = di.make_delta(_text2)
        self.assertTrue(di._has_index())
        self.assertEqual('N\x90/\x1fdiffer from\nagainst other text\n', delta)

    def test_add_source_max_bytes_to_index(self):
        if False:
            i = 10
            return i + 15
        di = self._gc_module.DeltaIndex()
        di._max_bytes_to_index = 3 * 16
        di.add_source(_text1, 0)
        di.add_source(_text3, 3)
        start2 = len(_text1) + 3
        (hash_list, entry_list) = di._dump_index()
        self.assertEqual(16, len(hash_list))
        self.assertEqual(67, len(entry_list))
        just_entries = sorted([(text_offset, hash_val) for (text_offset, hash_val) in entry_list if text_offset != 0 or hash_val != 0])
        rabin_hash = self._gc_module._rabin_hash
        self.assertEqual([(25, rabin_hash(_text1[10:26])), (50, rabin_hash(_text1[35:51])), (75, rabin_hash(_text1[60:76])), (start2 + 44, rabin_hash(_text3[29:45])), (start2 + 88, rabin_hash(_text3[73:89])), (start2 + 132, rabin_hash(_text3[117:133]))], just_entries)

    def test_second_add_source_triggers_make_index(self):
        if False:
            for i in range(10):
                print('nop')
        di = self._gc_module.DeltaIndex()
        self.assertFalse(di._has_index())
        di.add_source(_text1, 0)
        self.assertFalse(di._has_index())
        di.add_source(_text2, 0)
        self.assertTrue(di._has_index())

    def test_make_delta(self):
        if False:
            while True:
                i = 10
        di = self._gc_module.DeltaIndex(_text1)
        delta = di.make_delta(_text2)
        self.assertEqual('N\x90/\x1fdiffer from\nagainst other text\n', delta)

    def test_delta_against_multiple_sources(self):
        if False:
            i = 10
            return i + 15
        di = self._gc_module.DeltaIndex()
        di.add_source(_first_text, 0)
        self.assertEqual(len(_first_text), di._source_offset)
        di.add_source(_second_text, 0)
        self.assertEqual(len(_first_text) + len(_second_text), di._source_offset)
        delta = di.make_delta(_third_text)
        result = self._gc_module.apply_delta(_first_text + _second_text, delta)
        self.assertEqualDiff(_third_text, result)
        self.assertEqual('\x85\x01\x90\x14\x0chas some in \x91v6\x03and\x91d"\x91:\n', delta)

    def test_delta_with_offsets(self):
        if False:
            print('Hello World!')
        di = self._gc_module.DeltaIndex()
        di.add_source(_first_text, 5)
        self.assertEqual(len(_first_text) + 5, di._source_offset)
        di.add_source(_second_text, 10)
        self.assertEqual(len(_first_text) + len(_second_text) + 15, di._source_offset)
        delta = di.make_delta(_third_text)
        self.assertIsNot(None, delta)
        result = self._gc_module.apply_delta('12345' + _first_text + '1234567890' + _second_text, delta)
        self.assertIsNot(None, result)
        self.assertEqualDiff(_third_text, result)
        self.assertEqual('\x85\x01\x91\x05\x14\x0chas some in \x91\x856\x03and\x91s"\x91?\n', delta)

    def test_delta_with_delta_bytes(self):
        if False:
            return 10
        di = self._gc_module.DeltaIndex()
        source = _first_text
        di.add_source(_first_text, 0)
        self.assertEqual(len(_first_text), di._source_offset)
        delta = di.make_delta(_second_text)
        self.assertEqual('h\tsome more\x91\x019&previous text\nand has some extra text\n', delta)
        di.add_delta_source(delta, 0)
        source += delta
        self.assertEqual(len(_first_text) + len(delta), di._source_offset)
        second_delta = di.make_delta(_third_text)
        result = self._gc_module.apply_delta(source, second_delta)
        self.assertEqualDiff(_third_text, result)
        self.assertEqual('\x85\x01\x90\x14\x1chas some in common with the \x91S&\x03and\x91\x18,', second_delta)
        di.add_delta_source(second_delta, 0)
        source += second_delta
        third_delta = di.make_delta(_third_text)
        result = self._gc_module.apply_delta(source, third_delta)
        self.assertEqualDiff(_third_text, result)
        self.assertEqual('\x85\x01\x90\x14\x91~\x1c\x91S&\x03and\x91\x18,', third_delta)
        fourth_delta = di.make_delta(_fourth_text)
        self.assertEqual(_fourth_text, self._gc_module.apply_delta(source, fourth_delta))
        self.assertEqual('\x80\x01\x7f123456789012345\nsame rabin hash\n123456789012345\nsame rabin hash\n123456789012345\nsame rabin hash\n123456789012345\nsame rabin hash\x01\n', fourth_delta)
        di.add_delta_source(fourth_delta, 0)
        source += fourth_delta
        fifth_delta = di.make_delta(_fourth_text)
        self.assertEqual(_fourth_text, self._gc_module.apply_delta(source, fifth_delta))
        self.assertEqual('\x80\x01\x91§\x7f\x01\n', fifth_delta)

class TestCopyInstruction(tests.TestCase):

    def assertEncode(self, expected, offset, length):
        if False:
            print('Hello World!')
        bytes = _groupcompress_py.encode_copy_instruction(offset, length)
        if expected != bytes:
            self.assertEqual([hex(ord(e)) for e in expected], [hex(ord(b)) for b in bytes])

    def assertDecode(self, exp_offset, exp_length, exp_newpos, bytes, pos):
        if False:
            i = 10
            return i + 15
        cmd = ord(bytes[pos])
        pos += 1
        out = _groupcompress_py.decode_copy_instruction(bytes, cmd, pos)
        self.assertEqual((exp_offset, exp_length, exp_newpos), out)

    def test_encode_no_length(self):
        if False:
            return 10
        self.assertEncode('\x80', 0, 64 * 1024)
        self.assertEncode('\x81\x01', 1, 64 * 1024)
        self.assertEncode('\x81\n', 10, 64 * 1024)
        self.assertEncode('\x81ÿ', 255, 64 * 1024)
        self.assertEncode('\x82\x01', 256, 64 * 1024)
        self.assertEncode('\x83\x01\x01', 257, 64 * 1024)
        self.assertEncode('\x8fÿÿÿÿ', 4294967295, 64 * 1024)
        self.assertEncode('\x8eÿÿÿ', 4294967040, 64 * 1024)
        self.assertEncode('\x8dÿÿÿ', 4294902015, 64 * 1024)
        self.assertEncode('\x8bÿÿÿ', 4278255615, 64 * 1024)
        self.assertEncode('\x87ÿÿÿ', 16777215, 64 * 1024)
        self.assertEncode('\x8f\x04\x03\x02\x01', 16909060, 64 * 1024)

    def test_encode_no_offset(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEncode('\x90\x01', 0, 1)
        self.assertEncode('\x90\n', 0, 10)
        self.assertEncode('\x90ÿ', 0, 255)
        self.assertEncode('\xa0\x01', 0, 256)
        self.assertEncode('°\x01\x01', 0, 257)
        self.assertEncode('°ÿÿ', 0, 65535)
        self.assertEncode('\x80', 0, 64 * 1024)

    def test_encode(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEncode('\x91\x01\x01', 1, 1)
        self.assertEncode('\x91\t\n', 9, 10)
        self.assertEncode('\x91þÿ', 254, 255)
        self.assertEncode('¢\x02\x01', 512, 256)
        self.assertEncode('³\x02\x01\x01\x01', 258, 257)
        self.assertEncode('°\x01\x01', 0, 257)
        self.assertEncode('\x81\n', 10, 64 * 1024)

    def test_decode_no_length(self):
        if False:
            return 10
        self.assertDecode(0, 65536, 1, '\x80', 0)
        self.assertDecode(1, 65536, 2, '\x81\x01', 0)
        self.assertDecode(10, 65536, 2, '\x81\n', 0)
        self.assertDecode(255, 65536, 2, '\x81ÿ', 0)
        self.assertDecode(256, 65536, 2, '\x82\x01', 0)
        self.assertDecode(257, 65536, 3, '\x83\x01\x01', 0)
        self.assertDecode(4294967295, 65536, 5, '\x8fÿÿÿÿ', 0)
        self.assertDecode(4294967040, 65536, 4, '\x8eÿÿÿ', 0)
        self.assertDecode(4294902015, 65536, 4, '\x8dÿÿÿ', 0)
        self.assertDecode(4278255615, 65536, 4, '\x8bÿÿÿ', 0)
        self.assertDecode(16777215, 65536, 4, '\x87ÿÿÿ', 0)
        self.assertDecode(16909060, 65536, 5, '\x8f\x04\x03\x02\x01', 0)

    def test_decode_no_offset(self):
        if False:
            i = 10
            return i + 15
        self.assertDecode(0, 1, 2, '\x90\x01', 0)
        self.assertDecode(0, 10, 2, '\x90\n', 0)
        self.assertDecode(0, 255, 2, '\x90ÿ', 0)
        self.assertDecode(0, 256, 2, '\xa0\x01', 0)
        self.assertDecode(0, 257, 3, '°\x01\x01', 0)
        self.assertDecode(0, 65535, 3, '°ÿÿ', 0)
        self.assertDecode(0, 65536, 1, '\x80', 0)

    def test_decode(self):
        if False:
            print('Hello World!')
        self.assertDecode(1, 1, 3, '\x91\x01\x01', 0)
        self.assertDecode(9, 10, 3, '\x91\t\n', 0)
        self.assertDecode(254, 255, 3, '\x91þÿ', 0)
        self.assertDecode(512, 256, 3, '¢\x02\x01', 0)
        self.assertDecode(258, 257, 5, '³\x02\x01\x01\x01', 0)
        self.assertDecode(0, 257, 3, '°\x01\x01', 0)

    def test_decode_not_start(self):
        if False:
            print('Hello World!')
        self.assertDecode(1, 1, 6, 'abc\x91\x01\x01def', 3)
        self.assertDecode(9, 10, 5, 'ab\x91\t\nde', 2)
        self.assertDecode(254, 255, 6, 'not\x91þÿcopy', 3)

class TestBase128Int(tests.TestCase):
    scenarios = module_scenarios()
    _gc_module = None

    def assertEqualEncode(self, bytes, val):
        if False:
            print('Hello World!')
        self.assertEqual(bytes, self._gc_module.encode_base128_int(val))

    def assertEqualDecode(self, val, num_decode, bytes):
        if False:
            i = 10
            return i + 15
        self.assertEqual((val, num_decode), self._gc_module.decode_base128_int(bytes))

    def test_encode(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqualEncode('\x01', 1)
        self.assertEqualEncode('\x02', 2)
        self.assertEqualEncode('\x7f', 127)
        self.assertEqualEncode('\x80\x01', 128)
        self.assertEqualEncode('ÿ\x01', 255)
        self.assertEqualEncode('\x80\x02', 256)
        self.assertEqualEncode('ÿÿÿÿ\x0f', 4294967295)

    def test_decode(self):
        if False:
            print('Hello World!')
        self.assertEqualDecode(1, 1, '\x01')
        self.assertEqualDecode(2, 1, '\x02')
        self.assertEqualDecode(127, 1, '\x7f')
        self.assertEqualDecode(128, 2, '\x80\x01')
        self.assertEqualDecode(255, 2, 'ÿ\x01')
        self.assertEqualDecode(256, 2, '\x80\x02')
        self.assertEqualDecode(4294967295, 5, 'ÿÿÿÿ\x0f')

    def test_decode_with_trailing_bytes(self):
        if False:
            i = 10
            return i + 15
        self.assertEqualDecode(1, 1, '\x01abcdef')
        self.assertEqualDecode(127, 1, '\x7f\x01')
        self.assertEqualDecode(128, 2, '\x80\x01abcdef')
        self.assertEqualDecode(255, 2, 'ÿ\x01ÿ')