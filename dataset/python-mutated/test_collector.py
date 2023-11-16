from __future__ import annotations
from collections import defaultdict
import pprint
import unittest
from ansible.module_utils.facts import collector
from ansible.module_utils.facts import default_collectors

class TestFindCollectorsForPlatform(unittest.TestCase):

    def test(self):
        if False:
            return 10
        compat_platforms = [{'system': 'Generic'}]
        res = collector.find_collectors_for_platform(default_collectors.collectors, compat_platforms)
        for coll_class in res:
            self.assertIn(coll_class._platform, 'Generic')

    def test_linux(self):
        if False:
            return 10
        compat_platforms = [{'system': 'Linux'}]
        res = collector.find_collectors_for_platform(default_collectors.collectors, compat_platforms)
        for coll_class in res:
            self.assertIn(coll_class._platform, 'Linux')

    def test_linux_or_generic(self):
        if False:
            while True:
                i = 10
        compat_platforms = [{'system': 'Generic'}, {'system': 'Linux'}]
        res = collector.find_collectors_for_platform(default_collectors.collectors, compat_platforms)
        for coll_class in res:
            self.assertIn(coll_class._platform, ('Generic', 'Linux'))

class TestSelectCollectorNames(unittest.TestCase):

    def _assert_equal_detail(self, obj1, obj2, msg=None):
        if False:
            i = 10
            return i + 15
        msg = 'objects are not equal\n%s\n\n!=\n\n%s' % (pprint.pformat(obj1), pprint.pformat(obj2))
        return self.assertEqual(obj1, obj2, msg)

    def test(self):
        if False:
            i = 10
            return i + 15
        collector_names = ['distribution', 'all_ipv4_addresses', 'local', 'pkg_mgr']
        all_fact_subsets = self._all_fact_subsets()
        res = collector.select_collector_classes(collector_names, all_fact_subsets)
        expected = [default_collectors.DistributionFactCollector, default_collectors.PkgMgrFactCollector]
        self._assert_equal_detail(res, expected)

    def test_default_collectors(self):
        if False:
            return 10
        platform_info = {'system': 'Generic'}
        compat_platforms = [platform_info]
        collectors_for_platform = collector.find_collectors_for_platform(default_collectors.collectors, compat_platforms)
        (all_fact_subsets, aliases_map) = collector.build_fact_id_to_collector_map(collectors_for_platform)
        all_valid_subsets = frozenset(all_fact_subsets.keys())
        collector_names = collector.get_collector_names(valid_subsets=all_valid_subsets, aliases_map=aliases_map, platform_info=platform_info)
        complete_collector_names = collector._solve_deps(collector_names, all_fact_subsets)
        dep_map = collector.build_dep_data(complete_collector_names, all_fact_subsets)
        ordered_deps = collector.tsort(dep_map)
        ordered_collector_names = [x[0] for x in ordered_deps]
        res = collector.select_collector_classes(ordered_collector_names, all_fact_subsets)
        self.assertTrue(res.index(default_collectors.ServiceMgrFactCollector) > res.index(default_collectors.DistributionFactCollector), res)
        self.assertTrue(res.index(default_collectors.ServiceMgrFactCollector) > res.index(default_collectors.PlatformFactCollector), res)

    def _all_fact_subsets(self, data=None):
        if False:
            i = 10
            return i + 15
        all_fact_subsets = defaultdict(list)
        _data = {'pkg_mgr': [default_collectors.PkgMgrFactCollector], 'distribution': [default_collectors.DistributionFactCollector], 'network': [default_collectors.LinuxNetworkCollector]}
        data = data or _data
        for (key, value) in data.items():
            all_fact_subsets[key] = value
        return all_fact_subsets

class TestGetCollectorNames(unittest.TestCase):

    def test_none(self):
        if False:
            print('Hello World!')
        res = collector.get_collector_names()
        self.assertIsInstance(res, set)
        self.assertEqual(res, set([]))

    def test_empty_sets(self):
        if False:
            for i in range(10):
                print('nop')
        res = collector.get_collector_names(valid_subsets=frozenset([]), minimal_gather_subset=frozenset([]), gather_subset=[])
        self.assertIsInstance(res, set)
        self.assertEqual(res, set([]))

    def test_empty_valid_and_min_with_all_gather_subset(self):
        if False:
            return 10
        res = collector.get_collector_names(valid_subsets=frozenset([]), minimal_gather_subset=frozenset([]), gather_subset=['all'])
        self.assertIsInstance(res, set)
        self.assertEqual(res, set([]))

    def test_one_valid_with_all_gather_subset(self):
        if False:
            while True:
                i = 10
        valid_subsets = frozenset(['my_fact'])
        res = collector.get_collector_names(valid_subsets=valid_subsets, minimal_gather_subset=frozenset([]), gather_subset=['all'])
        self.assertIsInstance(res, set)
        self.assertEqual(res, set(['my_fact']))

    def _compare_res(self, gather_subset1, gather_subset2, valid_subsets=None, min_subset=None):
        if False:
            return 10
        valid_subsets = valid_subsets or frozenset()
        minimal_gather_subset = min_subset or frozenset()
        res1 = collector.get_collector_names(valid_subsets=valid_subsets, minimal_gather_subset=minimal_gather_subset, gather_subset=gather_subset1)
        res2 = collector.get_collector_names(valid_subsets=valid_subsets, minimal_gather_subset=minimal_gather_subset, gather_subset=gather_subset2)
        return (res1, res2)

    def test_not_all_other_order(self):
        if False:
            return 10
        valid_subsets = frozenset(['min_fact', 'something_else', 'whatever'])
        minimal_gather_subset = frozenset(['min_fact'])
        (res1, res2) = self._compare_res(['!all', 'whatever'], ['whatever', '!all'], valid_subsets=valid_subsets, min_subset=minimal_gather_subset)
        self.assertEqual(res1, res2)
        self.assertEqual(res1, set(['min_fact', 'whatever']))

    def test_not_all_other_order_min(self):
        if False:
            return 10
        valid_subsets = frozenset(['min_fact', 'something_else', 'whatever'])
        minimal_gather_subset = frozenset(['min_fact'])
        (res1, res2) = self._compare_res(['!min_fact', 'whatever'], ['whatever', '!min_fact'], valid_subsets=valid_subsets, min_subset=minimal_gather_subset)
        self.assertEqual(res1, res2)
        self.assertEqual(res1, set(['whatever']))

    def test_one_minimal_with_all_gather_subset(self):
        if False:
            while True:
                i = 10
        my_fact = 'my_fact'
        valid_subsets = frozenset([my_fact])
        minimal_gather_subset = valid_subsets
        res = collector.get_collector_names(valid_subsets=valid_subsets, minimal_gather_subset=minimal_gather_subset, gather_subset=['all'])
        self.assertIsInstance(res, set)
        self.assertEqual(res, set(['my_fact']))

    def test_with_all_gather_subset(self):
        if False:
            return 10
        valid_subsets = frozenset(['my_fact', 'something_else', 'whatever'])
        minimal_gather_subset = frozenset(['my_fact'])
        res = collector.get_collector_names(valid_subsets=valid_subsets, minimal_gather_subset=minimal_gather_subset, gather_subset=['all'])
        self.assertIsInstance(res, set)
        self.assertEqual(res, set(['my_fact', 'something_else', 'whatever']))

    def test_one_minimal_with_not_all_gather_subset(self):
        if False:
            for i in range(10):
                print('nop')
        valid_subsets = frozenset(['my_fact', 'something_else', 'whatever'])
        minimal_gather_subset = frozenset(['my_fact'])
        res = collector.get_collector_names(valid_subsets=valid_subsets, minimal_gather_subset=minimal_gather_subset, gather_subset=['!all'])
        self.assertIsInstance(res, set)
        self.assertEqual(res, set(['my_fact']))

    def test_gather_subset_excludes(self):
        if False:
            for i in range(10):
                print('nop')
        valid_subsets = frozenset(['my_fact', 'something_else', 'whatever'])
        minimal_gather_subset = frozenset(['min_fact', 'min_another'])
        res = collector.get_collector_names(valid_subsets=valid_subsets, minimal_gather_subset=minimal_gather_subset, gather_subset=['!min_fact', '!whatever'])
        self.assertIsInstance(res, set)
        self.assertEqual(res, set(['min_another']))

    def test_gather_subset_excludes_ordering(self):
        if False:
            for i in range(10):
                print('nop')
        valid_subsets = frozenset(['my_fact', 'something_else', 'whatever'])
        minimal_gather_subset = frozenset(['my_fact'])
        res = collector.get_collector_names(valid_subsets=valid_subsets, minimal_gather_subset=minimal_gather_subset, gather_subset=['!all', 'whatever'])
        self.assertIsInstance(res, set)
        self.assertEqual(res, set(['my_fact', 'whatever']))

    def test_gather_subset_excludes_min(self):
        if False:
            return 10
        valid_subsets = frozenset(['min_fact', 'something_else', 'whatever'])
        minimal_gather_subset = frozenset(['min_fact'])
        res = collector.get_collector_names(valid_subsets=valid_subsets, minimal_gather_subset=minimal_gather_subset, gather_subset=['whatever', '!min'])
        self.assertIsInstance(res, set)
        self.assertEqual(res, set(['whatever']))

    def test_gather_subset_excludes_min_and_all(self):
        if False:
            return 10
        valid_subsets = frozenset(['min_fact', 'something_else', 'whatever'])
        minimal_gather_subset = frozenset(['min_fact'])
        res = collector.get_collector_names(valid_subsets=valid_subsets, minimal_gather_subset=minimal_gather_subset, gather_subset=['whatever', '!all', '!min'])
        self.assertIsInstance(res, set)
        self.assertEqual(res, set(['whatever']))

    def test_invaid_gather_subset(self):
        if False:
            i = 10
            return i + 15
        valid_subsets = frozenset(['my_fact', 'something_else'])
        minimal_gather_subset = frozenset(['my_fact'])
        self.assertRaisesRegex(TypeError, 'Bad subset .* given to Ansible.*allowed\\:.*all,.*my_fact.*', collector.get_collector_names, valid_subsets=valid_subsets, minimal_gather_subset=minimal_gather_subset, gather_subset=['my_fact', 'not_a_valid_gather_subset'])

class TestFindUnresolvedRequires(unittest.TestCase):

    def test(self):
        if False:
            while True:
                i = 10
        names = ['network', 'virtual', 'env']
        all_fact_subsets = {'env': [default_collectors.EnvFactCollector], 'network': [default_collectors.LinuxNetworkCollector], 'virtual': [default_collectors.LinuxVirtualCollector]}
        res = collector.find_unresolved_requires(names, all_fact_subsets)
        self.assertIsInstance(res, set)
        self.assertEqual(res, set(['platform', 'distribution']))

    def test_resolved(self):
        if False:
            i = 10
            return i + 15
        names = ['network', 'virtual', 'env', 'platform', 'distribution']
        all_fact_subsets = {'env': [default_collectors.EnvFactCollector], 'network': [default_collectors.LinuxNetworkCollector], 'distribution': [default_collectors.DistributionFactCollector], 'platform': [default_collectors.PlatformFactCollector], 'virtual': [default_collectors.LinuxVirtualCollector]}
        res = collector.find_unresolved_requires(names, all_fact_subsets)
        self.assertIsInstance(res, set)
        self.assertEqual(res, set())

class TestBuildDepData(unittest.TestCase):

    def test(self):
        if False:
            print('Hello World!')
        names = ['network', 'virtual', 'env']
        all_fact_subsets = {'env': [default_collectors.EnvFactCollector], 'network': [default_collectors.LinuxNetworkCollector], 'virtual': [default_collectors.LinuxVirtualCollector]}
        res = collector.build_dep_data(names, all_fact_subsets)
        self.assertIsInstance(res, defaultdict)
        self.assertEqual(dict(res), {'network': set(['platform', 'distribution']), 'virtual': set(), 'env': set()})

class TestSolveDeps(unittest.TestCase):

    def test_no_solution(self):
        if False:
            print('Hello World!')
        unresolved = set(['required_thing1', 'required_thing2'])
        all_fact_subsets = {'env': [default_collectors.EnvFactCollector], 'network': [default_collectors.LinuxNetworkCollector], 'virtual': [default_collectors.LinuxVirtualCollector]}
        self.assertRaises(collector.CollectorNotFoundError, collector._solve_deps, unresolved, all_fact_subsets)

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        unresolved = set(['env', 'network'])
        all_fact_subsets = {'env': [default_collectors.EnvFactCollector], 'network': [default_collectors.LinuxNetworkCollector], 'virtual': [default_collectors.LinuxVirtualCollector], 'platform': [default_collectors.PlatformFactCollector], 'distribution': [default_collectors.DistributionFactCollector]}
        res = collector.resolve_requires(unresolved, all_fact_subsets)
        res = collector._solve_deps(unresolved, all_fact_subsets)
        self.assertIsInstance(res, set)
        for goal in unresolved:
            self.assertIn(goal, res)

class TestResolveRequires(unittest.TestCase):

    def test_no_resolution(self):
        if False:
            for i in range(10):
                print('nop')
        unresolved = ['required_thing1', 'required_thing2']
        all_fact_subsets = {'env': [default_collectors.EnvFactCollector], 'network': [default_collectors.LinuxNetworkCollector], 'virtual': [default_collectors.LinuxVirtualCollector]}
        self.assertRaisesRegex(collector.UnresolvedFactDep, 'unresolved fact dep.*required_thing2', collector.resolve_requires, unresolved, all_fact_subsets)

    def test(self):
        if False:
            i = 10
            return i + 15
        unresolved = ['env', 'network']
        all_fact_subsets = {'env': [default_collectors.EnvFactCollector], 'network': [default_collectors.LinuxNetworkCollector], 'virtual': [default_collectors.LinuxVirtualCollector]}
        res = collector.resolve_requires(unresolved, all_fact_subsets)
        for goal in unresolved:
            self.assertIn(goal, res)

    def test_exception(self):
        if False:
            return 10
        unresolved = ['required_thing1']
        all_fact_subsets = {}
        try:
            collector.resolve_requires(unresolved, all_fact_subsets)
        except collector.UnresolvedFactDep as exc:
            self.assertIn(unresolved[0], '%s' % exc)

class TestTsort(unittest.TestCase):

    def test(self):
        if False:
            while True:
                i = 10
        dep_map = {'network': set(['distribution', 'platform']), 'virtual': set(), 'platform': set(['what_platform_wants']), 'what_platform_wants': set(), 'network_stuff': set(['network'])}
        res = collector.tsort(dep_map)
        self.assertIsInstance(res, list)
        names = [x[0] for x in res]
        self.assertTrue(names.index('network_stuff') > names.index('network'))
        self.assertTrue(names.index('platform') > names.index('what_platform_wants'))
        self.assertTrue(names.index('network') > names.index('platform'))

    def test_cycles(self):
        if False:
            return 10
        dep_map = {'leaf1': set(), 'leaf2': set(), 'node1': set(['node2']), 'node2': set(['node3']), 'node3': set(['node1'])}
        self.assertRaises(collector.CycleFoundInFactDeps, collector.tsort, dep_map)

    def test_just_nodes(self):
        if False:
            while True:
                i = 10
        dep_map = {'leaf1': set(), 'leaf4': set(), 'leaf3': set(), 'leaf2': set()}
        res = collector.tsort(dep_map)
        self.assertIsInstance(res, list)
        names = [x[0] for x in res]
        self.assertEqual(set(names), set(dep_map.keys()))

    def test_self_deps(self):
        if False:
            while True:
                i = 10
        dep_map = {'node1': set(['node1']), 'node2': set(['node2'])}
        self.assertRaises(collector.CycleFoundInFactDeps, collector.tsort, dep_map)

    def test_unsolvable(self):
        if False:
            print('Hello World!')
        dep_map = {'leaf1': set(), 'node2': set(['leaf2'])}
        res = collector.tsort(dep_map)
        self.assertIsInstance(res, list)
        names = [x[0] for x in res]
        self.assertEqual(set(names), set(dep_map.keys()))

    def test_chain(self):
        if False:
            for i in range(10):
                print('nop')
        dep_map = {'leaf1': set(['leaf2']), 'leaf2': set(['leaf3']), 'leaf3': set(['leaf4']), 'leaf4': set(), 'leaf5': set(['leaf1'])}
        res = collector.tsort(dep_map)
        self.assertIsInstance(res, list)
        names = [x[0] for x in res]
        self.assertEqual(set(names), set(dep_map.keys()))

    def test_multi_pass(self):
        if False:
            print('Hello World!')
        dep_map = {'leaf1': set(), 'leaf2': set(['leaf3', 'leaf1', 'leaf4', 'leaf5']), 'leaf3': set(['leaf4', 'leaf1']), 'leaf4': set(['leaf1']), 'leaf5': set(['leaf1'])}
        res = collector.tsort(dep_map)
        self.assertIsInstance(res, list)
        names = [x[0] for x in res]
        self.assertEqual(set(names), set(dep_map.keys()))
        self.assertTrue(names.index('leaf1') < names.index('leaf2'))
        for leaf in ('leaf2', 'leaf3', 'leaf4', 'leaf5'):
            self.assertTrue(names.index('leaf1') < names.index(leaf))

class TestCollectorClassesFromGatherSubset(unittest.TestCase):
    maxDiff = None

    def _classes(self, all_collector_classes=None, valid_subsets=None, minimal_gather_subset=None, gather_subset=None, gather_timeout=None, platform_info=None):
        if False:
            while True:
                i = 10
        platform_info = platform_info or {'system': 'Linux'}
        return collector.collector_classes_from_gather_subset(all_collector_classes=all_collector_classes, valid_subsets=valid_subsets, minimal_gather_subset=minimal_gather_subset, gather_subset=gather_subset, gather_timeout=gather_timeout, platform_info=platform_info)

    def test_no_args(self):
        if False:
            for i in range(10):
                print('nop')
        res = self._classes()
        self.assertIsInstance(res, list)
        self.assertEqual(res, [])

    def test_not_all(self):
        if False:
            while True:
                i = 10
        res = self._classes(all_collector_classes=default_collectors.collectors, gather_subset=['!all'])
        self.assertIsInstance(res, list)
        self.assertEqual(res, [])

    def test_all(self):
        if False:
            return 10
        res = self._classes(all_collector_classes=default_collectors.collectors, gather_subset=['all'])
        self.assertIsInstance(res, list)

    def test_hardware(self):
        if False:
            i = 10
            return i + 15
        res = self._classes(all_collector_classes=default_collectors.collectors, gather_subset=['hardware'])
        self.assertIsInstance(res, list)
        self.assertIn(default_collectors.PlatformFactCollector, res)
        self.assertIn(default_collectors.LinuxHardwareCollector, res)
        self.assertTrue(res.index(default_collectors.LinuxHardwareCollector) > res.index(default_collectors.PlatformFactCollector))

    def test_network(self):
        if False:
            print('Hello World!')
        res = self._classes(all_collector_classes=default_collectors.collectors, gather_subset=['network'])
        self.assertIsInstance(res, list)
        self.assertIn(default_collectors.DistributionFactCollector, res)
        self.assertIn(default_collectors.PlatformFactCollector, res)
        self.assertIn(default_collectors.LinuxNetworkCollector, res)
        self.assertTrue(res.index(default_collectors.LinuxNetworkCollector) > res.index(default_collectors.PlatformFactCollector))
        self.assertTrue(res.index(default_collectors.LinuxNetworkCollector) > res.index(default_collectors.DistributionFactCollector))

    def test_env(self):
        if False:
            return 10
        res = self._classes(all_collector_classes=default_collectors.collectors, gather_subset=['env'])
        self.assertIsInstance(res, list)
        self.assertEqual(res, [default_collectors.EnvFactCollector])

    def test_facter(self):
        if False:
            return 10
        res = self._classes(all_collector_classes=default_collectors.collectors, gather_subset=set(['env', 'facter']))
        self.assertIsInstance(res, list)
        self.assertEqual(set(res), set([default_collectors.EnvFactCollector, default_collectors.FacterFactCollector]))

    def test_facter_ohai(self):
        if False:
            for i in range(10):
                print('nop')
        res = self._classes(all_collector_classes=default_collectors.collectors, gather_subset=set(['env', 'facter', 'ohai']))
        self.assertIsInstance(res, list)
        self.assertEqual(set(res), set([default_collectors.EnvFactCollector, default_collectors.FacterFactCollector, default_collectors.OhaiFactCollector]))

    def test_just_facter(self):
        if False:
            return 10
        res = self._classes(all_collector_classes=default_collectors.collectors, gather_subset=set(['facter']))
        self.assertIsInstance(res, list)
        self.assertEqual(set(res), set([default_collectors.FacterFactCollector]))

    def test_collector_specified_multiple_times(self):
        if False:
            print('Hello World!')
        res = self._classes(all_collector_classes=default_collectors.collectors, gather_subset=['platform', 'all', 'machine'])
        self.assertIsInstance(res, list)
        self.assertIn(default_collectors.PlatformFactCollector, res)

    def test_unknown_collector(self):
        if False:
            return 10
        self.assertRaisesRegex(TypeError, 'Bad subset.*unknown_collector.*given to Ansible.*allowed\\:.*all,.*env.*', self._classes, all_collector_classes=default_collectors.collectors, gather_subset=['env', 'unknown_collector'])