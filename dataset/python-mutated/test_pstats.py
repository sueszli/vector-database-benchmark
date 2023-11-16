import unittest
from test import support
from io import StringIO
from pstats import SortKey
import pstats
import cProfile

class AddCallersTestCase(unittest.TestCase):
    """Tests for pstats.add_callers helper."""

    def test_combine_results(self):
        if False:
            print('Hello World!')
        target = {'a': (1, 2, 3, 4)}
        source = {'a': (1, 2, 3, 4), 'b': (5, 6, 7, 8)}
        new_callers = pstats.add_callers(target, source)
        self.assertEqual(new_callers, {'a': (2, 4, 6, 8), 'b': (5, 6, 7, 8)})
        target = {'a': 1}
        source = {'a': 1, 'b': 5}
        new_callers = pstats.add_callers(target, source)
        self.assertEqual(new_callers, {'a': 2, 'b': 5})

class StatsTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        stats_file = support.findfile('pstats.pck')
        self.stats = pstats.Stats(stats_file)

    def test_add(self):
        if False:
            for i in range(10):
                print('nop')
        stream = StringIO()
        stats = pstats.Stats(stream=stream)
        stats.add(self.stats, self.stats)

    def test_sort_stats_int(self):
        if False:
            i = 10
            return i + 15
        valid_args = {-1: 'stdname', 0: 'calls', 1: 'time', 2: 'cumulative'}
        for (arg_int, arg_str) in valid_args.items():
            self.stats.sort_stats(arg_int)
            self.assertEqual(self.stats.sort_type, self.stats.sort_arg_dict_default[arg_str][-1])

    def test_sort_stats_string(self):
        if False:
            while True:
                i = 10
        for sort_name in ['calls', 'ncalls', 'cumtime', 'cumulative', 'filename', 'line', 'module', 'name', 'nfl', 'pcalls', 'stdname', 'time', 'tottime']:
            self.stats.sort_stats(sort_name)
            self.assertEqual(self.stats.sort_type, self.stats.sort_arg_dict_default[sort_name][-1])

    def test_sort_stats_partial(self):
        if False:
            while True:
                i = 10
        sortkey = 'filename'
        for sort_name in ['f', 'fi', 'fil', 'file', 'filen', 'filena', 'filenam', 'filename']:
            self.stats.sort_stats(sort_name)
            self.assertEqual(self.stats.sort_type, self.stats.sort_arg_dict_default[sortkey][-1])

    def test_sort_starts_mix(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, self.stats.sort_stats, 'calls', SortKey.TIME)
        self.assertRaises(TypeError, self.stats.sort_stats, SortKey.TIME, 'calls')

    def test_get_stats_profile(self):
        if False:
            while True:
                i = 10

        def pass1():
            if False:
                return 10
            pass

        def pass2():
            if False:
                i = 10
                return i + 15
            pass

        def pass3():
            if False:
                for i in range(10):
                    print('nop')
            pass
        pr = cProfile.Profile()
        pr.enable()
        pass1()
        pass2()
        pass3()
        pr.create_stats()
        ps = pstats.Stats(pr)
        stats_profile = ps.get_stats_profile()
        funcs_called = set(stats_profile.func_profiles.keys())
        self.assertIn('pass1', funcs_called)
        self.assertIn('pass2', funcs_called)
        self.assertIn('pass3', funcs_called)

    def test_SortKey_enum(self):
        if False:
            print('Hello World!')
        self.assertEqual(SortKey.FILENAME, 'filename')
        self.assertNotEqual(SortKey.FILENAME, SortKey.CALLS)
if __name__ == '__main__':
    unittest.main()