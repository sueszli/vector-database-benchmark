from concurrent.futures import ThreadPoolExecutor
import logging
import sys
import unittest
import unittest.mock
from coalib.settings.Section import Section
from coalib.core.Bear import Bear
from coalib.core.Core import initialize_dependencies, run
from coala_utils.decorators import generate_eq
from tests.core.CoreTestBase import CoreTestBase

@generate_eq('bear', 'section_name', 'file_dict')
class TestResult:

    def __init__(self, bear, section_name, file_dict):
        if False:
            i = 10
            return i + 15
        self.bear = bear
        self.section_name = section_name
        self.file_dict = file_dict

class TestBearBase(Bear):
    BEAR_DEPS = set()

    def analyze(self, bear, section_name, file_dict):
        if False:
            while True:
                i = 10
        return [TestResult(bear, section_name, file_dict)]

    def generate_tasks(self):
        if False:
            i = 10
            return i + 15
        return (((self, self.section.name, self.file_dict), {}),)

class CustomTasksBear(Bear):

    def __init__(self, section, file_dict, tasks=()):
        if False:
            return 10
        super().__init__(section, file_dict)
        self.tasks = tasks

    def analyze(self, *args):
        if False:
            return 10
        return args

    def generate_tasks(self):
        if False:
            i = 10
            return i + 15
        return ((task, {}) for task in self.tasks)

class BearA(TestBearBase):
    pass

class BearB(TestBearBase):
    pass

class BearC_NeedsB(TestBearBase):
    BEAR_DEPS = {BearB}

class BearD_NeedsC(TestBearBase):
    BEAR_DEPS = {BearC_NeedsB}

class BearE_NeedsAD(TestBearBase):
    BEAR_DEPS = {BearA, BearD_NeedsC}

class FailingBear(TestBearBase):

    def analyze(self, bear, section_name, file_dict):
        if False:
            return 10
        raise ValueError

class BearF_NeedsFailingBear(TestBearBase):
    BEAR_DEPS = {FailingBear}

class BearG_NeedsF(TestBearBase):
    BEAR_DEPS = {BearF_NeedsFailingBear}

class BearH_NeedsG(TestBearBase):
    BEAR_DEPS = {BearG_NeedsF}

class BearI_NeedsA_NeedsBDuringRuntime(TestBearBase):
    BEAR_DEPS = {BearA}

    def __init__(self, section, filedict):
        if False:
            print('Hello World!')
        super().__init__(section, filedict)
        self.BEAR_DEPS.add(BearB)

class BearJ_NeedsI(TestBearBase):
    BEAR_DEPS = {BearI_NeedsA_NeedsBDuringRuntime}

class BearK_NeedsA(TestBearBase):
    BEAR_DEPS = {BearA}

class BearL_NeedsA(TestBearBase):
    BEAR_DEPS = {BearA}

class MultiResultBear(TestBearBase):

    def analyze(self, bear, section_name, file_dict):
        if False:
            for i in range(10):
                print('nop')
        yield 1
        yield 2

class DynamicTaskBear(TestBearBase):
    BEAR_DEPS = {MultiResultBear, BearA}

    def analyze(self, run_id):
        if False:
            for i in range(10):
                print('nop')
        return [run_id]

    def generate_tasks(self):
        if False:
            while True:
                i = 10
        tasks_count = sum((len(results) for results in self.dependency_results.values()))
        return (((i,), {}) for i in range(tasks_count))
for i in range(100):
    classname = f'NoTasksBear{i}'
    generated_type = type(classname, (Bear,), dict(generate_tasks=lambda self: tuple()))
    setattr(sys.modules[__name__], classname, generated_type)

class DependentOnMultipleZeroTaskBearsTestBear(TestBearBase):
    BEAR_DEPS = {getattr(sys.modules[__name__], f'NoTasksBear{i}') for i in range(100)} | {MultiResultBear}

def get_next_instance(typ, iterable):
    if False:
        i = 10
        return i + 15
    '\n    Reads all elements in the iterable and returns the first occurrence\n    that is an instance of given type.\n\n    :param typ:\n        The type an object shall have.\n    :param iterable:\n        The iterable to search in.\n    :return:\n        The instance having ``typ`` or ``None`` if not found in\n        ``iterable``.\n    '
    try:
        return next((obj for obj in iterable if isinstance(obj, typ)))
    except StopIteration:
        return None

class InitializeDependenciesTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.section1 = Section('test-section1')
        self.section2 = Section('test-section2')
        self.filedict1 = {'f1': []}
        self.filedict2 = {'f2': []}

    def test_multi_dependencies(self):
        if False:
            return 10
        bear_e = BearE_NeedsAD(self.section1, self.filedict1)
        (dependency_tracker, bears_to_schedule) = initialize_dependencies({bear_e})
        self.assertEqual(len(dependency_tracker.get_dependencies(bear_e)), 2)
        self.assertTrue(any((isinstance(bear, BearA) for bear in dependency_tracker.get_dependencies(bear_e))))
        self.assertTrue(any((isinstance(bear, BearD_NeedsC) for bear in dependency_tracker.get_dependencies(bear_e))))
        bear_a = get_next_instance(BearA, dependency_tracker.get_dependencies(bear_e))
        self.assertIsNotNone(bear_a)
        self.assertIs(bear_a.section, self.section1)
        self.assertIs(bear_a.file_dict, self.filedict1)
        self.assertEqual(dependency_tracker.get_dependencies(bear_a), set())
        bear_d = get_next_instance(BearD_NeedsC, dependency_tracker.get_dependencies(bear_e))
        self.assertIsNotNone(bear_d)
        self.assertIs(bear_d.section, self.section1)
        self.assertIs(bear_d.file_dict, self.filedict1)
        self.assertEqual(len(dependency_tracker.get_dependencies(bear_d)), 1)
        self.assertEqual(len(dependency_tracker.get_dependencies(bear_d)), 1)
        bear_c = dependency_tracker.get_dependencies(bear_d).pop()
        self.assertIs(bear_c.section, self.section1)
        self.assertIs(bear_c.file_dict, self.filedict1)
        self.assertIsInstance(bear_c, BearC_NeedsB)
        self.assertEqual(len(dependency_tracker.get_dependencies(bear_c)), 1)
        bear_b = dependency_tracker.get_dependencies(bear_c).pop()
        self.assertIs(bear_b.section, self.section1)
        self.assertIs(bear_b.file_dict, self.filedict1)
        self.assertIsInstance(bear_b, BearB)
        self.assertEqual(dependency_tracker.get_dependencies(bear_b), set())
        self.assertEqual(bears_to_schedule, {bear_a, bear_b})

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        bear_a = BearA(self.section1, self.filedict1)
        bear_b = BearB(self.section1, self.filedict1)
        (dependency_tracker, bears_to_schedule) = initialize_dependencies({bear_a, bear_b})
        self.assertTrue(dependency_tracker.are_dependencies_resolved)
        self.assertEqual(bears_to_schedule, {bear_a, bear_b})

    def test_reuse_instantiated_dependencies(self):
        if False:
            while True:
                i = 10
        bear_b = BearB(self.section1, self.filedict1)
        bear_c = BearC_NeedsB(self.section1, self.filedict1)
        (dependency_tracker, bears_to_schedule) = initialize_dependencies({bear_b, bear_c})
        self.assertEqual(dependency_tracker.dependants, {bear_c})
        self.assertEqual(dependency_tracker.get_dependencies(bear_c), {bear_b})
        self.assertEqual(bears_to_schedule, {bear_b})

    def test_no_reuse_of_different_section_dependency(self):
        if False:
            while True:
                i = 10
        bear_b = BearB(self.section1, self.filedict1)
        bear_c = BearC_NeedsB(self.section2, self.filedict1)
        (dependency_tracker, bears_to_schedule) = initialize_dependencies({bear_b, bear_c})
        self.assertEqual(dependency_tracker.dependants, {bear_c})
        dependencies = dependency_tracker.dependencies
        self.assertEqual(len(dependencies), 1)
        dependency = dependencies.pop()
        self.assertIsInstance(dependency, BearB)
        self.assertIsNot(dependency, bear_b)
        self.assertEqual(bears_to_schedule, {bear_b, dependency})

    def test_different_sections_different_dependency_instances(self):
        if False:
            while True:
                i = 10
        bear_c_section1 = BearC_NeedsB(self.section1, self.filedict1)
        bear_c_section2 = BearC_NeedsB(self.section2, self.filedict1)
        (dependency_tracker, bears_to_schedule) = initialize_dependencies({bear_c_section1, bear_c_section2})
        bear_c_s1_dependencies = dependency_tracker.get_dependencies(bear_c_section1)
        self.assertEqual(len(bear_c_s1_dependencies), 1)
        bear_b_section1 = bear_c_s1_dependencies.pop()
        self.assertIsInstance(bear_b_section1, BearB)
        bear_c_s2_dependencies = dependency_tracker.get_dependencies(bear_c_section2)
        self.assertEqual(len(bear_c_s2_dependencies), 1)
        bear_b_section2 = bear_c_s2_dependencies.pop()
        self.assertIsInstance(bear_b_section2, BearB)
        self.assertIsNot(bear_b_section1, bear_b_section2)
        self.assertEqual(bears_to_schedule, {bear_b_section1, bear_b_section2})

    def test_reuse_multiple_same_dependencies_correctly(self):
        if False:
            i = 10
            return i + 15
        bear_c = BearC_NeedsB(self.section1, self.filedict1)
        bear_b1 = BearB(self.section1, self.filedict1)
        bear_b2 = BearB(self.section1, self.filedict1)
        (dependency_tracker, bears_to_schedule) = initialize_dependencies({bear_c, bear_b1, bear_b2})
        bear_c_dependencies = dependency_tracker.get_dependencies(bear_c)
        self.assertEqual(len(bear_c_dependencies), 1)
        bear_c_dependency = bear_c_dependencies.pop()
        self.assertIsInstance(bear_c_dependency, BearB)
        self.assertIn(bear_c_dependency, {bear_b1, bear_b2})
        self.assertEqual(bears_to_schedule, {bear_b1, bear_b2})

    def test_correct_reuse_of_implicitly_instantiated_dependency(self):
        if False:
            print('Hello World!')
        bear_c1 = BearC_NeedsB(self.section1, self.filedict1)
        bear_c2 = BearC_NeedsB(self.section1, self.filedict1)
        (dependency_tracker, bears_to_schedule) = initialize_dependencies({bear_c1, bear_c2})
        bear_c1_dependencies = dependency_tracker.get_dependencies(bear_c1)
        self.assertEqual(len(bear_c1_dependencies), 1)
        bear_b1 = bear_c1_dependencies.pop()
        self.assertIsInstance(bear_b1, BearB)
        bear_c2_dependencies = dependency_tracker.get_dependencies(bear_c2)
        self.assertEqual(len(bear_c2_dependencies), 1)
        bear_b2 = bear_c2_dependencies.pop()
        self.assertIsInstance(bear_b2, BearB)
        self.assertIs(bear_b1, bear_b2)

    def test_empty_case(self):
        if False:
            print('Hello World!')
        (dependency_tracker, bears_to_schedule) = initialize_dependencies(set())
        self.assertTrue(dependency_tracker.are_dependencies_resolved)
        self.assertEqual(bears_to_schedule, set())

    def test_different_filedict_different_dependency_instance(self):
        if False:
            for i in range(10):
                print('nop')
        bear_b = BearB(self.section1, self.filedict1)
        bear_c = BearC_NeedsB(self.section1, self.filedict2)
        (dependency_tracker, bears_to_schedule) = initialize_dependencies({bear_b, bear_c})
        self.assertEqual(dependency_tracker.dependants, {bear_c})
        dependencies = dependency_tracker.dependencies
        self.assertEqual(len(dependencies), 1)
        dependency = dependencies.pop()
        self.assertIsInstance(dependency, BearB)
        self.assertIsNot(dependency, bear_b)
        self.assertEqual(bears_to_schedule, {bear_b, dependency})

    def test_out_of_order_grouping(self):
        if False:
            i = 10
            return i + 15
        sections = [Section('test-section' + str(i)) for i in range(1000)]
        bears_c = [BearC_NeedsB(section, self.filedict1) for section in sections]
        bears_b = [BearB(section, self.filedict1) for section in sections]
        (dependency_tracker, bears_to_schedule) = initialize_dependencies(set(bears_c) | set(bears_b))
        self.assertEqual(set(dependency_tracker), set(zip(bears_b, bears_c)))
        self.assertEqual(bears_to_schedule, set(bears_b))

class CoreTest(CoreTestBase):

    def setUp(self):
        if False:
            return 10
        logging.getLogger().setLevel(logging.DEBUG)
        self.section1 = Section('test-section1')
        self.filedict1 = {'f1': []}

    @staticmethod
    def get_comparable_results(results):
        if False:
            print('Hello World!')
        '\n        Transforms an iterable of ``TestResult`` into something comparable.\n\n        Some ``TestResult`` instances returned by ``run`` contain instance\n        values. Due to the ``ProcessPoolExecutor``, objects get pickled,\n        are transferred to the other process and are re-instantiated,\n        effectively changing the id of them. The same happens again on the\n        transfer back in the results, so we need something that can be\n        compared.\n\n        This function extracts relevant values into a tuple, containing::\n\n            (test_result.bear.name,\n             test_result.section_name,\n             test_result.file_dict)\n\n        :param results:\n            The results to transform.\n        :return:\n            A list of comparable results for tests.\n        '
        return [(result.bear.name, result.section_name, result.file_dict) for result in results]

    def assertTestResultsEqual(self, real, expected):
        if False:
            return 10
        '\n        Test whether results from ``execute_run`` do equal with the ones given.\n\n        This function does a sequence comparison without order, so for example\n        ``[1, 2, 1]`` and ``[2, 1, 1]`` are considered equal.\n\n        :param real:\n            The actual results.\n        :param expected:\n            The expected results.\n        '
        comparable_real = self.get_comparable_results(real)
        self.assertEqual(len(comparable_real), len(expected))
        for result in expected:
            self.assertIn(result, comparable_real)
            comparable_real.remove(result)

    def test_run_simple(self):
        if False:
            while True:
                i = 10
        bear = CustomTasksBear(self.section1, self.filedict1, tasks=[(0,)])
        results = self.execute_run({bear})
        self.assertEqual(results, [0])
        self.assertEqual(bear.dependency_results, {})

    def test_run_complex(self):
        if False:
            while True:
                i = 10
        bear_e = BearE_NeedsAD(self.section1, self.filedict1)
        results = self.execute_run({bear_e})
        self.assertTestResultsEqual(results, [(BearE_NeedsAD.name, self.section1.name, self.filedict1)])
        self.assertEqual(results[-1].bear.name, bear_e.name)
        self.assertIn(BearA, bear_e.dependency_results)
        self.assertIn(BearD_NeedsC, bear_e.dependency_results)
        self.assertEqual(len(bear_e.dependency_results), 2)
        self.assertTestResultsEqual(bear_e.dependency_results[BearA], [(BearA.name, self.section1.name, self.filedict1)])
        self.assertTestResultsEqual(bear_e.dependency_results[BearD_NeedsC], [(BearD_NeedsC.name, self.section1.name, self.filedict1)])
        bear_a = get_next_instance(BearA, (result.bear for result in bear_e.dependency_results[BearA]))
        self.assertIsNotNone(bear_a)
        self.assertEqual(bear_a.dependency_results, {})
        bear_d = get_next_instance(BearD_NeedsC, (result.bear for result in bear_e.dependency_results[BearD_NeedsC]))
        self.assertIsNotNone(bear_d)
        self.assertIn(BearC_NeedsB, bear_d.dependency_results)
        self.assertEqual(len(bear_d.dependency_results), 1)
        self.assertTestResultsEqual(bear_d.dependency_results[BearC_NeedsB], [(BearC_NeedsB.name, self.section1.name, self.filedict1)])
        bear_c = get_next_instance(BearC_NeedsB, (result.bear for result in bear_d.dependency_results[BearC_NeedsB]))
        self.assertIsNotNone(bear_c)
        self.assertIn(BearB, bear_c.dependency_results)
        self.assertEqual(len(bear_c.dependency_results), 1)
        self.assertTestResultsEqual(bear_c.dependency_results[BearB], [(BearB.name, self.section1.name, self.filedict1)])
        bear_b = get_next_instance(BearB, (result.bear for result in bear_c.dependency_results[BearB]))
        self.assertIsNotNone(bear_b)
        self.assertEqual(bear_b.dependency_results, {})

    def test_run_multiple_bears(self):
        if False:
            print('Hello World!')
        bear1 = BearA(self.section1, self.filedict1)
        bear2 = BearB(self.section1, self.filedict1)
        results = self.execute_run({bear1, bear2})
        self.assertTestResultsEqual(results, [(BearA.name, self.section1.name, self.filedict1), (BearB.name, self.section1.name, self.filedict1)])

    def test_run_multiple_bears_with_independent_dependencies(self):
        if False:
            for i in range(10):
                print('nop')
        bear1 = BearK_NeedsA(self.section1, self.filedict1)
        bear2 = BearC_NeedsB(self.section1, self.filedict1)
        results = self.execute_run({bear1, bear2})
        self.assertTestResultsEqual(results, [(BearK_NeedsA.name, self.section1.name, self.filedict1), (BearC_NeedsB.name, self.section1.name, self.filedict1)])
        self.assertEqual(len(bear1.dependency_results), 1)
        self.assertTestResultsEqual(bear1.dependency_results[BearA], [(BearA.name, self.section1.name, self.filedict1)])
        self.assertEqual(len(bear2.dependency_results), 1)
        self.assertTestResultsEqual(bear2.dependency_results[BearB], [(BearB.name, self.section1.name, self.filedict1)])

    def test_run_multiple_bears_with_same_dependencies(self):
        if False:
            i = 10
            return i + 15
        bear1 = BearK_NeedsA(self.section1, self.filedict1)
        bear2 = BearL_NeedsA(self.section1, self.filedict1)
        results = self.execute_run({bear1, bear2})
        self.assertTestResultsEqual(results, [(BearK_NeedsA.name, self.section1.name, self.filedict1), (BearL_NeedsA.name, self.section1.name, self.filedict1)])
        self.assertEqual(len(bear1.dependency_results), 1)
        self.assertEqual(bear1.dependency_results[BearA], bear2.dependency_results[BearA])
        self.assertTestResultsEqual(bear1.dependency_results[BearA], [(BearA.name, self.section1.name, self.filedict1)])

    def test_run_same_bear_twice(self):
        if False:
            print('Hello World!')
        bear1 = BearA(self.section1, self.filedict1)
        bear2 = BearA(self.section1, self.filedict1)
        results = self.execute_run({bear1, bear2})
        self.assertTestResultsEqual(results, [(BearA.name, self.section1.name, self.filedict1), (BearA.name, self.section1.name, self.filedict1)])

    def test_run_dependency_bear_explicitly(self):
        if False:
            i = 10
            return i + 15
        bear = BearD_NeedsC(self.section1, self.filedict1)
        bear_dependency = BearB(self.section1, self.filedict1)
        results = self.execute_run({bear, bear_dependency})
        self.assertTestResultsEqual(results, [(BearB.name, self.section1.name, self.filedict1), (BearD_NeedsC.name, self.section1.name, self.filedict1)])
        self.assertEqual(len(bear.dependency_results), 1)
        self.assertTestResultsEqual(bear.dependency_results[BearC_NeedsB], [(BearC_NeedsB.name, self.section1.name, self.filedict1)])
        bear_c = get_next_instance(BearC_NeedsB, (result.bear for result in bear.dependency_results[BearC_NeedsB]))
        self.assertEqual(len(bear_c.dependency_results), 1)
        self.assertTestResultsEqual(bear_c.dependency_results[BearB], [(BearB.name, self.section1.name, self.filedict1)])

    def test_run_result_handler_exception(self):
        if False:
            for i in range(10):
                print('nop')
        bear = CustomTasksBear(self.section1, self.filedict1, tasks=[(x,) for x in range(10)])
        on_result = unittest.mock.Mock(side_effect=ValueError)
        with self.assertLogs(logging.getLogger()) as cm:
            run({bear}, on_result)
        on_result.assert_has_calls([unittest.mock.call(i) for i in range(10)], any_order=True)
        self.assertEqual(len(cm.output), 10)
        for i in range(10):
            self.assertTrue(cm.output[i].startswith('ERROR:root:An exception was thrown during result-handling.'))

    def test_run_bear_exception(self):
        if False:
            print('Hello World!')
        with self.assertLogs(logging.getLogger()) as cm:
            results = self.execute_run({FailingBear(self.section1, self.filedict1)})
        self.assertEqual(results, [])
        self.assertEqual(len(cm.output), 1)
        self.assertTrue(cm.output[0].startswith('ERROR:root:An exception was thrown during bear execution.'))

    def test_run_bear_exception_with_other_bears(self):
        if False:
            while True:
                i = 10
        with self.assertLogs(logging.getLogger()) as cm:
            results = self.execute_run({FailingBear(self.section1, self.filedict1), CustomTasksBear(self.section1, self.filedict1, tasks=[(x,) for x in range(3)])})
        self.assertEqual(len(cm.output), 1)
        self.assertTrue(cm.output[0].startswith('ERROR:root:An exception was thrown during bear execution.'))
        self.assertEqual(set(results), {0, 1, 2})

    def test_run_bear_with_multiple_tasks(self):
        if False:
            for i in range(10):
                print('nop')
        bear = CustomTasksBear(self.section1, self.filedict1, tasks=[(x,) for x in range(3)])
        results = self.execute_run({bear})
        result_set = set(results)
        self.assertEqual(len(result_set), len(results))
        self.assertEqual(result_set, {0, 1, 2})
        self.assertEqual(bear.dependency_results, {})

    def test_run_bear_exception_with_dependencies(self):
        if False:
            print('Hello World!')
        bear_a = BearA(self.section1, self.filedict1)
        bear_failing = BearH_NeedsG(self.section1, self.filedict1)
        results = self.execute_run({bear_a, bear_failing})
        self.assertTestResultsEqual(results, [(BearA.name, self.section1.name, self.filedict1)])
        self.assertEqual(bear_a.dependency_results, {})
        self.assertEqual(bear_failing.dependency_results, {})

    def test_run_bear_with_0_tasks(self):
        if False:
            while True:
                i = 10
        bear = CustomTasksBear(self.section1, self.filedict1, tasks=[])
        results = self.execute_run({bear})
        self.assertEqual(len(results), 0)
        self.assertEqual(bear.dependency_results, {})

    def test_run_generate_tasks_dynamically_with_dependency_results(self):
        if False:
            print('Hello World!')
        bear = DynamicTaskBear(self.section1, self.filedict1)
        results = self.execute_run({bear})
        self.assertEqual(len(results), 3)
        self.assertEqual(len(bear.dependency_results), 2)
        self.assertIn(MultiResultBear, bear.dependency_results)
        self.assertIn(BearA, bear.dependency_results)
        self.assertEqual(len(bear.dependency_results[MultiResultBear]), 2)
        self.assertEqual(len(bear.dependency_results[BearA]), 1)

    def test_run_multiple_dependency_bears_with_zero_tasks(self):
        if False:
            return 10
        uut = DependentOnMultipleZeroTaskBearsTestBear(self.section1, self.filedict1)
        results = self.execute_run({uut})
        self.assertEqual(len(results), 1)
        uut_result = get_next_instance(TestResult, results)
        self.assertEqual(uut_result.bear.name, uut.name)
        self.assertEqual(uut_result.section_name, self.section1.name)
        self.assertEqual(uut_result.file_dict, self.filedict1)
        self.assertEqual(len(uut.dependency_results), 1)
        self.assertEqual(uut.dependency_results[MultiResultBear], [1, 2])

    def test_run_heavy_cpu_load(self):
        if False:
            while True:
                i = 10
        bear = CustomTasksBear(self.section1, self.filedict1, tasks=[(x,) for x in range(100)])
        results = self.execute_run({bear})
        result_set = set(results)
        self.assertEqual(len(result_set), len(results))
        self.assertEqual(result_set, set(range(100)))
        self.assertEqual(bear.dependency_results, {})

    def test_run_empty(self):
        if False:
            print('Hello World!')
        self.execute_run(set())

    def test_bears_with_runtime_dependencies(self):
        if False:
            return 10
        bear = BearI_NeedsA_NeedsBDuringRuntime(self.section1, self.filedict1)
        results = self.execute_run({bear})
        self.assertTestResultsEqual(results, [(BearI_NeedsA_NeedsBDuringRuntime.name, self.section1.name, self.filedict1)])
        self.assertEqual(len(bear.dependency_results), 2)
        self.assertTestResultsEqual(bear.dependency_results[BearA], [(BearA.name, self.section1.name, self.filedict1)])
        self.assertTestResultsEqual(bear.dependency_results[BearB], [(BearB.name, self.section1.name, self.filedict1)])
        bear = BearJ_NeedsI(self.section1, self.filedict1)
        results = self.execute_run({bear})
        self.assertTestResultsEqual(results, [(BearJ_NeedsI.name, self.section1.name, self.filedict1)])
        self.assertEqual(len(bear.dependency_results), 1)
        self.assertTestResultsEqual(bear.dependency_results[BearI_NeedsA_NeedsBDuringRuntime], [(BearI_NeedsA_NeedsBDuringRuntime.name, self.section1.name, self.filedict1)])

class CoreOnThreadPoolExecutorTest(CoreTest):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.executor = (ThreadPoolExecutor, tuple(), dict(max_workers=8))

class CoreOnSpecificExecutorTest(CoreTestBase):

    def test_custom_executor_closed_after_run(self):
        if False:
            return 10
        bear = CustomTasksBear(Section('test-section'), {'some-file': []}, tasks=[(0,)])
        for bears in [set(), {bear}]:
            executor = ThreadPoolExecutor(max_workers=1)
            self.execute_run(bears, executor=executor)
            with self.assertRaisesRegex(RuntimeError, 'cannot schedule new futures after shutdown'):
                executor.submit(lambda : None)

class CoreCacheTest(CoreTestBase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.executor = (ThreadPoolExecutor, tuple(), dict(max_workers=1))

    def test_no_cache(self):
        if False:
            for i in range(10):
                print('nop')
        section = Section('test-section')
        filedict = {}
        task_args = (3, 4, 5)
        bear = CustomTasksBear(section, filedict, tasks=[task_args])
        with unittest.mock.patch.object(bear, 'analyze', wraps=bear.analyze) as mock:
            results = self.execute_run({bear})
            mock.assert_called_once_with(*task_args)
            self.assertEqual(results, list(task_args))
            mock.reset_mock()
            results = self.execute_run({bear})
            mock.assert_called_once_with(*task_args)
            self.assertEqual(results, list(task_args))
            mock.reset_mock()
            results = self.execute_run({bear}, None)
            mock.assert_called_once_with(*task_args)
            self.assertEqual(results, list(task_args))

    def test_cache(self):
        if False:
            for i in range(10):
                print('nop')
        section = Section('test-section')
        filedict = {}
        cache = {}
        task_args = (10, 11, 12)
        bear = CustomTasksBear(section, filedict, tasks=[task_args])
        with unittest.mock.patch.object(bear, 'analyze', wraps=bear.analyze) as mock:
            results = self.execute_run({bear}, cache)
            mock.assert_called_once_with(*task_args)
            self.assertEqual(results, list(task_args))
            self.assertEqual(len(cache), 1)
            self.assertEqual(next(iter(cache.keys())), CustomTasksBear)
            self.assertEqual(len(next(iter(cache.values()))), 1)
            for i in range(3):
                mock.reset_mock()
                results = self.execute_run({bear}, cache)
                self.assertFalse(mock.called)
                self.assertEqual(results, list(task_args))
                self.assertEqual(len(cache), 1)
                self.assertIn(CustomTasksBear, cache)
                self.assertEqual(len(next(iter(cache.values()))), 1)
        task_args = (500, 11, 12)
        bear = CustomTasksBear(section, filedict, tasks=[task_args])
        with unittest.mock.patch.object(bear, 'analyze', wraps=bear.analyze) as mock:
            results = self.execute_run({bear}, cache)
            mock.assert_called_once_with(*task_args)
            self.assertEqual(results, list(task_args))
            self.assertEqual(len(cache), 1)
            self.assertIn(CustomTasksBear, cache)
            self.assertEqual(len(next(iter(cache.values()))), 2)
            mock.reset_mock()
            results = self.execute_run({bear}, cache)
            self.assertFalse(mock.called)
            self.assertEqual(results, list(task_args))
            self.assertEqual(len(cache), 1)
            self.assertIn(CustomTasksBear, cache)
            self.assertEqual(len(next(iter(cache.values()))), 2)

    def test_existing_cache_with_unrelated_data(self):
        if False:
            print('Hello World!')
        section = Section('test-section')
        filedict = {}
        cache = {CustomTasksBear: {b'123456': [100, 101, 102]}}
        task_args = (-1, -2, -3)
        bear = CustomTasksBear(section, filedict, tasks=[task_args])
        with unittest.mock.patch.object(bear, 'analyze', wraps=bear.analyze) as mock:
            results = self.execute_run({bear}, cache)
            mock.assert_called_once_with(*task_args)
            self.assertEqual(results, list(task_args))
            self.assertEqual(len(cache), 1)
            self.assertIn(CustomTasksBear, cache)
            cache_values = next(iter(cache.values()))
            self.assertEqual(len(cache_values), 2)
            self.assertIn(b'123456', cache_values)
            self.assertEqual(cache_values[b'123456'], [100, 101, 102])