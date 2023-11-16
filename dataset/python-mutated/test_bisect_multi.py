"""Tests for bisect_multi."""
from bzrlib.bisect_multi import bisect_multi_bytes
from bzrlib.tests import TestCase

class TestBisectMultiBytes(TestCase):

    def test_lookup_no_keys_no_calls(self):
        if False:
            i = 10
            return i + 15
        calls = []

        def missing_content(location_keys):
            if False:
                for i in range(10):
                    print('nop')
            calls.append(location_keys)
            return ((location_key, False) for location_key in location_keys)
        self.assertEqual([], list(bisect_multi_bytes(missing_content, 100, [])))
        self.assertEqual([], calls)

    def test_lookup_missing_key_no_content(self):
        if False:
            return 10
        "Doing a lookup in a zero-length file still does a single request.\n\n        This makes sense because the bisector cannot tell how long content is\n        and its more flexible to only stop when the content object says 'False'\n        for a given location, key pair.\n        "
        calls = []

        def missing_content(location_keys):
            if False:
                while True:
                    i = 10
            calls.append(location_keys)
            return ((location_key, False) for location_key in location_keys)
        self.assertEqual([], list(bisect_multi_bytes(missing_content, 0, ['foo', 'bar'])))
        self.assertEqual([[(0, 'foo'), (0, 'bar')]], calls)

    def test_lookup_missing_key_before_all_others(self):
        if False:
            print('Hello World!')
        calls = []

        def missing_first_content(location_keys):
            if False:
                for i in range(10):
                    print('nop')
            calls.append(location_keys)
            result = []
            for location_key in location_keys:
                if location_key[0] == 0:
                    result.append((location_key, False))
                else:
                    result.append((location_key, -1))
            return result
        self.assertEqual([], list(bisect_multi_bytes(missing_first_content, 0, ['foo', 'bar'])))
        self.assertEqual([[(0, 'foo'), (0, 'bar')]], calls)
        del calls[:]
        self.assertEqual([], list(bisect_multi_bytes(missing_first_content, 2, ['foo', 'bar'])))
        self.assertEqual([[(1, 'foo'), (1, 'bar')], [(0, 'foo'), (0, 'bar')]], calls)
        del calls[:]
        self.assertEqual([], list(bisect_multi_bytes(missing_first_content, 268435456 - 1, ['foo', 'bar'])))
        self.assertEqual([[(134217727, 'foo'), (134217727, 'bar')], [(67108864, 'foo'), (67108864, 'bar')], [(33554433, 'foo'), (33554433, 'bar')], [(16777218, 'foo'), (16777218, 'bar')], [(8388611, 'foo'), (8388611, 'bar')], [(4194308, 'foo'), (4194308, 'bar')], [(2097157, 'foo'), (2097157, 'bar')], [(1048582, 'foo'), (1048582, 'bar')], [(524295, 'foo'), (524295, 'bar')], [(262152, 'foo'), (262152, 'bar')], [(131081, 'foo'), (131081, 'bar')], [(65546, 'foo'), (65546, 'bar')], [(32779, 'foo'), (32779, 'bar')], [(16396, 'foo'), (16396, 'bar')], [(8205, 'foo'), (8205, 'bar')], [(4110, 'foo'), (4110, 'bar')], [(2063, 'foo'), (2063, 'bar')], [(1040, 'foo'), (1040, 'bar')], [(529, 'foo'), (529, 'bar')], [(274, 'foo'), (274, 'bar')], [(147, 'foo'), (147, 'bar')], [(84, 'foo'), (84, 'bar')], [(53, 'foo'), (53, 'bar')], [(38, 'foo'), (38, 'bar')], [(31, 'foo'), (31, 'bar')], [(28, 'foo'), (28, 'bar')], [(27, 'foo'), (27, 'bar')], [(26, 'foo'), (26, 'bar')], [(25, 'foo'), (25, 'bar')], [(24, 'foo'), (24, 'bar')], [(23, 'foo'), (23, 'bar')], [(22, 'foo'), (22, 'bar')], [(21, 'foo'), (21, 'bar')], [(20, 'foo'), (20, 'bar')], [(19, 'foo'), (19, 'bar')], [(18, 'foo'), (18, 'bar')], [(17, 'foo'), (17, 'bar')], [(16, 'foo'), (16, 'bar')], [(15, 'foo'), (15, 'bar')], [(14, 'foo'), (14, 'bar')], [(13, 'foo'), (13, 'bar')], [(12, 'foo'), (12, 'bar')], [(11, 'foo'), (11, 'bar')], [(10, 'foo'), (10, 'bar')], [(9, 'foo'), (9, 'bar')], [(8, 'foo'), (8, 'bar')], [(7, 'foo'), (7, 'bar')], [(6, 'foo'), (6, 'bar')], [(5, 'foo'), (5, 'bar')], [(4, 'foo'), (4, 'bar')], [(3, 'foo'), (3, 'bar')], [(2, 'foo'), (2, 'bar')], [(1, 'foo'), (1, 'bar')], [(0, 'foo'), (0, 'bar')]], calls)

    def test_lookup_missing_key_after_all_others(self):
        if False:
            print('Hello World!')
        calls = []
        end = None

        def missing_last_content(location_keys):
            if False:
                i = 10
                return i + 15
            calls.append(location_keys)
            result = []
            for location_key in location_keys:
                if location_key[0] == end:
                    result.append((location_key, False))
                else:
                    result.append((location_key, +1))
            return result
        end = 0
        self.assertEqual([], list(bisect_multi_bytes(missing_last_content, 0, ['foo', 'bar'])))
        self.assertEqual([[(0, 'foo'), (0, 'bar')]], calls)
        del calls[:]
        end = 2
        self.assertEqual([], list(bisect_multi_bytes(missing_last_content, 3, ['foo', 'bar'])))
        self.assertEqual([[(1, 'foo'), (1, 'bar')], [(2, 'foo'), (2, 'bar')]], calls)
        del calls[:]
        end = 268435456 - 2
        self.assertEqual([], list(bisect_multi_bytes(missing_last_content, 268435456 - 1, ['foo', 'bar'])))
        self.assertEqual([[(134217727, 'foo'), (134217727, 'bar')], [(201326590, 'foo'), (201326590, 'bar')], [(234881021, 'foo'), (234881021, 'bar')], [(251658236, 'foo'), (251658236, 'bar')], [(260046843, 'foo'), (260046843, 'bar')], [(264241146, 'foo'), (264241146, 'bar')], [(266338297, 'foo'), (266338297, 'bar')], [(267386872, 'foo'), (267386872, 'bar')], [(267911159, 'foo'), (267911159, 'bar')], [(268173302, 'foo'), (268173302, 'bar')], [(268304373, 'foo'), (268304373, 'bar')], [(268369908, 'foo'), (268369908, 'bar')], [(268402675, 'foo'), (268402675, 'bar')], [(268419058, 'foo'), (268419058, 'bar')], [(268427249, 'foo'), (268427249, 'bar')], [(268431344, 'foo'), (268431344, 'bar')], [(268433391, 'foo'), (268433391, 'bar')], [(268434414, 'foo'), (268434414, 'bar')], [(268434925, 'foo'), (268434925, 'bar')], [(268435180, 'foo'), (268435180, 'bar')], [(268435307, 'foo'), (268435307, 'bar')], [(268435370, 'foo'), (268435370, 'bar')], [(268435401, 'foo'), (268435401, 'bar')], [(268435416, 'foo'), (268435416, 'bar')], [(268435423, 'foo'), (268435423, 'bar')], [(268435426, 'foo'), (268435426, 'bar')], [(268435427, 'foo'), (268435427, 'bar')], [(268435428, 'foo'), (268435428, 'bar')], [(268435429, 'foo'), (268435429, 'bar')], [(268435430, 'foo'), (268435430, 'bar')], [(268435431, 'foo'), (268435431, 'bar')], [(268435432, 'foo'), (268435432, 'bar')], [(268435433, 'foo'), (268435433, 'bar')], [(268435434, 'foo'), (268435434, 'bar')], [(268435435, 'foo'), (268435435, 'bar')], [(268435436, 'foo'), (268435436, 'bar')], [(268435437, 'foo'), (268435437, 'bar')], [(268435438, 'foo'), (268435438, 'bar')], [(268435439, 'foo'), (268435439, 'bar')], [(268435440, 'foo'), (268435440, 'bar')], [(268435441, 'foo'), (268435441, 'bar')], [(268435442, 'foo'), (268435442, 'bar')], [(268435443, 'foo'), (268435443, 'bar')], [(268435444, 'foo'), (268435444, 'bar')], [(268435445, 'foo'), (268435445, 'bar')], [(268435446, 'foo'), (268435446, 'bar')], [(268435447, 'foo'), (268435447, 'bar')], [(268435448, 'foo'), (268435448, 'bar')], [(268435449, 'foo'), (268435449, 'bar')], [(268435450, 'foo'), (268435450, 'bar')], [(268435451, 'foo'), (268435451, 'bar')], [(268435452, 'foo'), (268435452, 'bar')], [(268435453, 'foo'), (268435453, 'bar')], [(268435454, 'foo'), (268435454, 'bar')]], calls)

    def test_lookup_when_a_key_is_missing_continues(self):
        if False:
            return 10
        calls = []

        def missing_foo_otherwise_missing_first_content(location_keys):
            if False:
                for i in range(10):
                    print('nop')
            calls.append(location_keys)
            result = []
            for location_key in location_keys:
                if location_key[1] == 'foo' or location_key[0] == 0:
                    result.append((location_key, False))
                else:
                    result.append((location_key, -1))
            return result
        self.assertEqual([], list(bisect_multi_bytes(missing_foo_otherwise_missing_first_content, 2, ['foo', 'bar'])))
        self.assertEqual([[(1, 'foo'), (1, 'bar')], [(0, 'bar')]], calls)

    def test_found_keys_returned_other_searches_continue(self):
        if False:
            i = 10
            return i + 15
        calls = []

        def find_bar_at_1_foo_missing_at_0(location_keys):
            if False:
                return 10
            calls.append(location_keys)
            result = []
            for location_key in location_keys:
                if location_key == (1, 'bar'):
                    result.append((location_key, 'bar-result'))
                elif location_key[0] == 0:
                    result.append((location_key, False))
                else:
                    result.append((location_key, -1))
            return result
        self.assertEqual([('bar', 'bar-result')], list(bisect_multi_bytes(find_bar_at_1_foo_missing_at_0, 4, ['foo', 'bar'])))
        self.assertEqual([[(2, 'foo'), (2, 'bar')], [(1, 'foo'), (1, 'bar')], [(0, 'foo')]], calls)

    def test_searches_different_keys_in_different_directions(self):
        if False:
            i = 10
            return i + 15
        calls = []

        def missing_bar_at_1_foo_at_3(location_keys):
            if False:
                return 10
            calls.append(location_keys)
            result = []
            for location_key in location_keys:
                if location_key[1] == 'bar':
                    if location_key[0] == 1:
                        result.append((location_key, False))
                    else:
                        result.append((location_key, -1))
                elif location_key[1] == 'foo':
                    if location_key[0] == 3:
                        result.append((location_key, False))
                    else:
                        result.append((location_key, +1))
            return result
        self.assertEqual([], list(bisect_multi_bytes(missing_bar_at_1_foo_at_3, 4, ['foo', 'bar'])))
        self.assertEqual([[(2, 'foo'), (2, 'bar')], [(3, 'foo'), (1, 'bar')]], calls)

    def test_change_direction_in_single_key_search(self):
        if False:
            i = 10
            return i + 15
        calls = []

        def missing_at_5(location_keys):
            if False:
                print('Hello World!')
            calls.append(location_keys)
            result = []
            for location_key in location_keys:
                if location_key[0] == 5:
                    result.append((location_key, False))
                elif location_key[0] > 5:
                    result.append((location_key, -1))
                else:
                    result.append((location_key, +1))
            return result
        self.assertEqual([], list(bisect_multi_bytes(missing_at_5, 8, ['foo', 'bar'])))
        self.assertEqual([[(4, 'foo'), (4, 'bar')], [(6, 'foo'), (6, 'bar')], [(5, 'foo'), (5, 'bar')]], calls)