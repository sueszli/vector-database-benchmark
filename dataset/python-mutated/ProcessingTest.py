import copy
import logging
import multiprocessing
import os
import platform
import queue
import subprocess
import sys
import unittest
from pyprint.ConsolePrinter import ConsolePrinter
from testfixtures import LogCapture, StringComparison
from coalib.bears.Bear import Bear
from coalib.output.printers.LogPrinter import LogPrinter
from coalib.output.printers.ListLogPrinter import ListLogPrinter
from coalib.processes.CONTROL_ELEMENT import CONTROL_ELEMENT
from coalib.processes.Processing import ACTIONS, autoapply_actions, check_result_ignore, create_process_group, execute_section, get_default_actions, get_file_dict, print_result, process_queues, simplify_section_result, yield_ignore_ranges, instantiate_bears
from coalib.results.HiddenResult import HiddenResult
from coalib.results.Result import RESULT_SEVERITY, Result
from coalib.results.result_actions.ApplyPatchAction import ApplyPatchAction
from coalib.results.result_actions.PrintDebugMessageAction import PrintDebugMessageAction
from coalib.results.result_actions.ResultAction import ResultAction
from coalib.results.SourceRange import SourceRange
from coalib.settings.ConfigurationGathering import gather_configuration
from coalib.settings.Section import Section
from coalib.settings.Setting import Setting
from coalib.misc.Caching import FileCache
process_group_test_code = '\nimport time, subprocess, os, platform, sys;\np=subprocess.Popen([sys.executable,\n                  "-c",\n                  "import time; time.sleep(0.1)"]);\npgid = p.pid if platform.system() == "Windows" else os.getpgid(p.pid);\nprint(p.pid, pgid)\np.terminate()\n'

class DummyProcess(multiprocessing.Process):

    def __init__(self, control_queue, starts_dead=False):
        if False:
            for i in range(10):
                print('nop')
        multiprocessing.Process.__init__(self)
        self.control_queue = control_queue
        self.starts_dead = starts_dead

    def is_alive(self):
        if False:
            i = 10
            return i + 15
        return not self.control_queue.empty() and (not self.starts_dead)

class ProcessingTestLogPrinter(LogPrinter):

    def __init__(self, log_queue):
        if False:
            return 10
        LogPrinter.__init__(self, self)
        self.log_queue = log_queue
        self.set_up = False

    def log_message(self, log_message, timestamp=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.log_queue.put(log_message)

class ProcessingTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'section_executor_test_files', '.coafile'))
        self.testcode_c_path = os.path.join(os.path.dirname(config_path), 'testcode.c')
        self.unreadable_path = os.path.join(os.path.dirname(config_path), 'unreadable')
        factory_test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'file_factory_test_files'))
        self.factory_test_file = os.path.join(factory_test_path, 'factory_test.txt')
        self.a_bear_test_path = os.path.join(factory_test_path, 'a_bear_test.txt')
        self.b_bear_test_path = os.path.join(factory_test_path, 'b_bear_test.txt')
        self.c_bear_test_path = os.path.join(factory_test_path, 'c_bear_test.txt')
        self.d_bear_test_path = os.path.join(factory_test_path, 'd_bear_test.txt')
        self.e_bear_test_path = os.path.join(factory_test_path, 'e_bear_test.txt')
        self.n_bear_test_path = os.path.join(factory_test_path, 'n_bear_test.txt')
        self.n_bear_test_path_2 = os.path.join(factory_test_path, 'n_bear_test2.txt')
        self.x_bear_test_path = os.path.join(factory_test_path, 'x_bear_test.txt')
        filename_list = [self.factory_test_file, self.a_bear_test_path, self.b_bear_test_path, self.c_bear_test_path, self.d_bear_test_path, self.e_bear_test_path, self.n_bear_test_path, self.n_bear_test_path_2, self.x_bear_test_path]
        self.file_dict = get_file_dict(filename_list)
        self.result_queue = queue.Queue()
        self.queue = queue.Queue()
        self.log_queue = queue.Queue()
        self.console_printer = ConsolePrinter()
        log_printer = LogPrinter(ConsolePrinter())
        self.log_printer = ProcessingTestLogPrinter(self.log_queue)
        (self.sections, self.local_bears, self.global_bears, targets) = gather_configuration(lambda *args: True, log_printer, arg_list=['--config', config_path])
        self.assertEqual(len(self.local_bears['cli']), 1)
        self.assertEqual(len(self.global_bears['cli']), 1)
        self.assertEqual(targets, [])

    def test_run(self):
        if False:
            while True:
                i = 10
        self.sections['cli'].append(Setting('jobs', '1'))
        cache = FileCache(self.log_printer, 'coala_test', flush_cache=True)
        results = execute_section(self.sections['cli'], self.global_bears['cli'], self.local_bears['cli'], lambda *args: self.result_queue.put(args[2]), cache, self.log_printer, console_printer=self.console_printer)
        self.assertTrue(results[0])
        local_results = self.result_queue.get(timeout=0)
        global_results = self.result_queue.get(timeout=0)
        self.assertTrue(self.result_queue.empty())
        self.assertEqual(len(local_results), 1)
        self.assertEqual(len(global_results), 1)
        self.assertEqual(len(results[1]), 1)
        self.assertEqual(len(results[2]), 1)
        local_result = local_results[0]
        global_result = global_results[0]
        self.assertRegex(repr(local_result), f"<Result object\\(id={hex(local_result.id)}, origin='LocalTestBear', affected_code=\\(\\), severity=NORMAL, confidence=100, message='test msg', aspect=NoneType, applied_actions=\\{{\\}}\\) at 0x[0-9a-fA-F]+>")
        self.assertRegex(repr(global_result), f"<Result object\\(id={hex(global_result.id)}, origin='GlobalTestBear', affected_code=\\(.*start=.*file=.*section_executor_test_files.*line=None.*end=.*\\), severity=NORMAL, confidence=100, message='test message', aspect=NoneType, applied_actions=\\{{\\}}\\) at 0x[0-9a-fA-F]+>")

    def test_empty_run(self):
        if False:
            print('Hello World!')
        execute_section(self.sections['cli'], [], [], lambda *args: self.result_queue.put(args[2]), None, self.log_printer, console_printer=self.console_printer)
        self.sections['cli'].append(Setting('jobs', 'bogus!'))
        results = execute_section(self.sections['cli'], [], [], lambda *args: self.result_queue.put(args[2]), None, self.log_printer, console_printer=self.console_printer)
        self.assertFalse(results[0])
        self.assertEqual(len(results[1]), 1)
        self.assertEqual(len(results[2]), 0)

    def test_mixed_run(self):
        if False:
            for i in range(10):
                print('nop')
        self.sections['mixed'].append(Setting('jobs', '1'))
        log_printer = ListLogPrinter()
        global_bears = self.global_bears['mixed']
        local_bears = self.local_bears['mixed']
        bears = global_bears + local_bears
        with LogCapture() as capture:
            execute_section(self.sections['mixed'], global_bears, local_bears, lambda *args: self.result_queue.put(args[2]), None, log_printer, console_printer=self.console_printer)
        capture.check(('root', 'ERROR', "Bears that uses raw files can't be mixed with Bears that uses text files. Please move the following bears to their own section: " + ', '.join((bear.name for bear in bears if not bear.USE_RAW_FILES))))

    def test_raw_run(self):
        if False:
            while True:
                i = 10
        self.sections['raw'].append(Setting('jobs', '1'))
        results = execute_section(self.sections['raw'], self.global_bears['raw'], self.local_bears['raw'], lambda *args: self.result_queue.put(args[2]), None, self.log_printer, console_printer=self.console_printer)
        self.assertTrue(results[0])
        self.assertEqual(len(results[1]), 1)
        self.assertEqual(len(results[2]), 1)
        self.assertEqual(self.unreadable_path.lower(), results[1].keys()[0].lower())
        self.unreadable_path = results[1].keys()[0]
        self.assertEqual([bear.name for bear in self.global_bears['raw']], results[2].keys())
        self.assertEqual(results[1][self.unreadable_path], [Result('LocalTestRawBear', 'test msg')])
        self.assertEqual(results[2][self.global_bears['raw'][0].name], [Result.from_values('GlobalTestRawBear', 'test message', self.unreadable_path)])

    def test_process_queues(self):
        if False:
            return 10
        ctrlq = queue.Queue()
        ctrlq.put((CONTROL_ELEMENT.LOCAL, 1))
        ctrlq.put((CONTROL_ELEMENT.LOCAL_FINISHED, None))
        ctrlq.put((CONTROL_ELEMENT.GLOBAL, 1))
        ctrlq.put((CONTROL_ELEMENT.LOCAL, 2))
        ctrlq.put((CONTROL_ELEMENT.GLOBAL_FINISHED, None))
        ctrlq.put((CONTROL_ELEMENT.LOCAL_FINISHED, None))
        ctrlq.put((CONTROL_ELEMENT.GLOBAL, 1))
        ctrlq.put((CONTROL_ELEMENT.GLOBAL_FINISHED, None))
        first_local = Result.from_values('o', 'The first result.', file='f')
        second_local = Result.from_values('ABear', 'The second result.', file='f', line=1)
        third_local = Result.from_values('ABear', 'The second result.', file='f', line=4)
        fourth_local = Result.from_values('ABear', 'Another result.', file='f', line=7)
        first_global = Result('o', 'The one and only global result.')
        section = Section('')
        section.append(Setting('min_severity', 'normal'))
        process_queues([DummyProcess(control_queue=ctrlq) for i in range(3)], ctrlq, {1: [first_local, second_local, third_local, Result('o', 'm', severity=RESULT_SEVERITY.INFO), Result.from_values('ABear', 'u', 'f', 2, 1), Result.from_values('ABear', 'u', 'f', 3, 1)], 2: [fourth_local, HiddenResult('t', 'c'), Result.from_values('ABear', 'u', 'f', 5, 1), Result.from_values('ABear', 'u', 'f', 6, 1)]}, {1: [first_global]}, {'f': self.file_dict[self.factory_test_file]}, lambda *args: self.queue.put(args[2]), section, None, self.log_printer, self.console_printer)
        self.assertEqual(self.queue.get(timeout=0), [second_local, third_local])
        self.assertEqual(self.queue.get(timeout=0), [fourth_local])
        self.assertEqual(self.queue.get(timeout=0), [first_global])
        self.assertEqual(self.queue.get(timeout=0), [first_global])

    def test_dead_processes(self):
        if False:
            return 10
        ctrlq = queue.Queue()
        ctrlq.put((CONTROL_ELEMENT.GLOBAL_FINISHED, None))
        ctrlq.put((CONTROL_ELEMENT.LOCAL_FINISHED, None))
        process_queues([DummyProcess(ctrlq, starts_dead=True) for i in range(3)], ctrlq, {}, {}, {}, lambda *args: self.queue.put(args[2]), Section(''), None, self.log_printer, self.console_printer)
        with self.assertRaises(queue.Empty):
            self.queue.get(timeout=0)
        ctrlq.put((CONTROL_ELEMENT.LOCAL_FINISHED, None))
        ctrlq.put((CONTROL_ELEMENT.GLOBAL_FINISHED, None))
        process_queues([DummyProcess(ctrlq, starts_dead=True) for i in range(3)], ctrlq, {}, {}, {}, lambda *args: self.queue.put(args[2]), Section(''), None, self.log_printer, self.console_printer)
        with self.assertRaises(queue.Empty):
            self.queue.get(timeout=0)

    def test_create_process_group(self):
        if False:
            while True:
                i = 10
        p = create_process_group([sys.executable, '-c', process_group_test_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        retval = p.wait()
        if retval != 0:
            for line in p.stderr:
                print(line, end='')
            raise Exception('Subprocess did not exit correctly')
        output = [i for i in p.stdout]
        p.stderr.close()
        p.stdout.close()
        (pid, pgid) = [int(i.strip()) for i_out in output for i in i_out.split()]
        if platform.system() != 'Windows':
            self.assertEqual(p.pid, pgid)

    def test_get_file_dict(self):
        if False:
            i = 10
            return i + 15
        file_dict = get_file_dict([self.testcode_c_path], self.log_printer)
        self.assertEqual(len(file_dict), 1)
        self.assertEqual(type(file_dict[self.testcode_c_path]), tuple, msg='files in file_dict should not be editable')

    def test_get_file_dict_non_existent_file(self):
        if False:
            while True:
                i = 10
        with LogCapture() as capture:
            file_dict = get_file_dict(['non_existent_file'], self.log_printer)
        self.assertEqual(file_dict, {})
        capture.check(('root', 'WARNING', StringComparison(".*Failed to read file 'non_existent_file' because of an unknown error.*")), ('root', 'INFO', StringComparison('.*Exception was:.*')))

    def test_get_file_dict_allow_raw_file(self):
        if False:
            i = 10
            return i + 15
        file_dict = get_file_dict([self.unreadable_path], self.log_printer, True)
        self.assertNotEqual(file_dict, {})
        self.assertEqual(file_dict[self.unreadable_path], None)

    def test_get_file_dict_forbid_raw_file(self):
        if False:
            for i in range(10):
                print('nop')
        log_printer = ListLogPrinter()
        with LogCapture() as capture:
            file_dict = get_file_dict([self.unreadable_path], log_printer, False)
        self.assertEqual(file_dict, {})
        capture.check(('root', 'WARNING', f"Failed to read file '{self.unreadable_path}'. It seems to contain non-unicode characters. Leaving it out."))

    def test_simplify_section_result(self):
        if False:
            print('Hello World!')
        results = (True, {'file1': [Result('a', 'b')], 'file2': None}, {'file3': [Result('a', 'c')]}, None)
        (yielded, yielded_unfixed, all_results) = simplify_section_result(results)
        self.assertEqual(yielded, True)
        self.assertEqual(yielded_unfixed, True)
        self.assertEqual(len(all_results), 2)

    def test_ignore_results(self):
        if False:
            return 10
        ranges = [([], SourceRange.from_values('f', 1, 1, 2, 2))]
        result = Result.from_values('origin (Something Specific)', 'message', file='e', line=1, column=1, end_line=2, end_column=2)
        self.assertFalse(check_result_ignore(result, ranges))
        ranges.append(([], SourceRange.from_values('e', 2, 3, 3, 3)))
        self.assertFalse(check_result_ignore(result, ranges))
        ranges.append(([], SourceRange.from_values('e', 1, 1, 2, 2)))
        self.assertTrue(check_result_ignore(result, ranges))
        result1 = Result.from_values('origin', 'message', file='e')
        self.assertTrue(check_result_ignore(result1, ranges))
        ranges = [(['something', 'else', 'not origin'], SourceRange.from_values('e', 1, 1, 2, 2))]
        self.assertFalse(check_result_ignore(result, ranges))
        ranges = [(['something', 'else', 'origin'], SourceRange.from_values('e', 1, 1, 2, 2))]
        self.assertTrue(check_result_ignore(result, ranges))

    def test_ignore_glob(self):
        if False:
            while True:
                i = 10
        result = Result.from_values('LineLengthBear', 'message', file='d', line=1, column=1, end_line=2, end_column=2)
        ranges = [(['(line*|space*)', 'py*'], SourceRange.from_values('d', 1, 1, 2, 2))]
        self.assertTrue(check_result_ignore(result, ranges))
        result = Result.from_values('SpaceConsistencyBear', 'message', file='d', line=1, column=1, end_line=2, end_column=2)
        ranges = [(['(line*|space*)', 'py*'], SourceRange.from_values('d', 1, 1, 2, 2))]
        self.assertTrue(check_result_ignore(result, ranges))
        result = Result.from_values('XMLBear', 'message', file='d', line=1, column=1, end_line=2, end_column=2)
        ranges = [(['(line*|space*)', 'py*'], SourceRange.from_values('d', 1, 1, 2, 2))]
        self.assertFalse(check_result_ignore(result, ranges))

    def test_yield_ignore_ranges(self):
        if False:
            for i in range(10):
                print('nop')
        test_file_dict_a = {'f': self.file_dict[self.a_bear_test_path]}
        test_ignore_range_a = list(yield_ignore_ranges(test_file_dict_a))
        for (test_bears, test_source_range) in test_ignore_range_a:
            self.assertEqual(test_bears, ['abear'])
            self.assertEqual(test_source_range.start.line, 1)
            self.assertEqual(test_source_range.start.column, 1)
            self.assertEqual(test_source_range.end.line, 2)
            self.assertEqual(test_source_range.end.column, 43)
        test_file_dict_b = {'f': self.file_dict[self.b_bear_test_path]}
        test_ignore_range_b = list(yield_ignore_ranges(test_file_dict_b))
        for (test_bears, test_source_range) in test_ignore_range_b:
            self.assertEqual(test_bears, ['bbear'])
            self.assertEqual(test_source_range.start.line, 1)
            self.assertEqual(test_source_range.start.column, 1)
            self.assertEqual(test_source_range.end.line, 3)
            self.assertEqual(test_source_range.end.column, 16)
        test_file_dict_c = {'f': self.file_dict[self.c_bear_test_path]}
        test_ignore_range_c = list(yield_ignore_ranges(test_file_dict_c))
        for (test_bears, test_source_range) in test_ignore_range_c:
            self.assertEqual(test_bears, ['cbear'])
            self.assertEqual(test_source_range.start.line, 1)
            self.assertEqual(test_source_range.start.column, 1)
            self.assertEqual(test_source_range.end.line, 2)
            self.assertEqual(test_source_range.end.column, 42)
        test_file_dict_d = {'f': self.file_dict[self.d_bear_test_path]}
        test_ignore_range_d = list(yield_ignore_ranges(test_file_dict_d))
        for (test_bears, test_source_range) in test_ignore_range_d:
            self.assertEqual(test_bears, ['cbear'])
            self.assertEqual(test_source_range.start.line, 1)
            self.assertEqual(test_source_range.start.column, 1)
            self.assertEqual(test_source_range.end.line, 2)
            self.assertEqual(test_source_range.end.column, 20)
        test_file_dict_e = {'f': self.file_dict[self.e_bear_test_path]}
        test_ignore_range_e = list(yield_ignore_ranges(test_file_dict_e))
        for (test_bears, test_source_range) in test_ignore_range_e:
            self.assertEqual(test_bears, [])
            self.assertEqual(test_source_range.start.line, 1)
            self.assertEqual(test_source_range.start.column, 1)
            self.assertEqual(test_source_range.end.line, 2)
            self.assertEqual(test_source_range.end.column, 43)
        test_file_dict_n = {'f': self.file_dict[self.n_bear_test_path]}
        test_ignore_range_n = list(yield_ignore_ranges(test_file_dict_n))
        for (test_bears, test_source_range) in test_ignore_range_n:
            self.assertEqual(test_bears, ['nbear'])
            self.assertEqual(test_source_range.start.line, 1)
            self.assertEqual(test_source_range.start.column, 1)
            self.assertEqual(test_source_range.end.line, 2)
            self.assertEqual(test_source_range.end.column, 43)
        test_file_dict_n = {'f': self.file_dict[self.n_bear_test_path_2]}
        test_ignore_range_n = list(yield_ignore_ranges(test_file_dict_n))
        for (test_bears, test_source_range) in test_ignore_range_n:
            self.assertEqual(test_bears, [])
            self.assertEqual(test_source_range.start.line, 1)
            self.assertEqual(test_source_range.start.column, 1)
            self.assertEqual(test_source_range.end.line, 2)
            self.assertEqual(test_source_range.end.column, 43)
        test_file_dict_single_line = {'f': self.file_dict[self.x_bear_test_path]}
        test_ignore_range_single_line = list(yield_ignore_ranges(test_file_dict_single_line))
        self.assertEqual(len(test_ignore_range_single_line), 1)
        (bears, source_range) = test_ignore_range_single_line[0]
        self.assertEqual(bears, ['xbear'])
        self.assertEqual(source_range.start.line, 1)
        self.assertEqual(source_range.start.column, 1)
        self.assertEqual(source_range.end.line, 1)
        self.assertEqual(source_range.end.column, 15)

    def test_loaded_bears_with_error_result(self):
        if False:
            i = 10
            return i + 15

        class BearWithMissingPrerequisites(Bear):

            def __init__(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                Bear.__init__(self, *args, **kwargs)

            def run(self):
                if False:
                    return 10
                return []

            @classmethod
            def check_prerequisites(cls):
                if False:
                    return 10
                return False
        multiprocessing.Queue()
        tmp_local_bears = copy.copy(self.local_bears['cli'])
        tmp_local_bears.append(BearWithMissingPrerequisites)
        cache = FileCache(self.log_printer, 'coala_test_on_error', flush_cache=True)
        results = execute_section(self.sections['cli'], [], tmp_local_bears, lambda *args: self.result_queue.put(args[2]), cache, self.log_printer, console_printer=self.console_printer)
        self.assertEqual(len(cache.data), 0)
        cache = FileCache(self.log_printer, 'coala_test_on_error', flush_cache=False)
        results = execute_section(self.sections['cli'], [], self.local_bears['cli'], lambda *args: self.result_queue.put(args[2]), cache, self.log_printer, console_printer=self.console_printer)
        self.assertGreater(len(cache.data), 0)

    def test_global_instantiation(self):
        if False:
            return 10

        class TestOneBear(Bear):

            def __init__(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                raise RuntimeError

        class TestTwoBear(Bear):
            BEAR_DEPS = {TestOneBear}

            def __init__(self, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                raise RuntimeError

        class TestThreeBear(Bear):
            BEAR_DEPS = {TestTwoBear}

            def __init__(self, file_dict, section, queue, timeout=0.1):
                if False:
                    print('Hello World!')
                Bear.__init__(self, section, queue, timeout)
        global_bear_list = [TestTwoBear, TestThreeBear]
        (list1, list2) = instantiate_bears(self.sections['cli'], [], global_bear_list, {}, self.queue, console_printer=self.console_printer, debug=False)
        self.assertEqual(len(list1), 0)
        self.assertEqual(len(list2), 1)
        with self.assertRaises(RuntimeError):
            global_bear_list = [TestOneBear]
            instantiate_bears(self.sections['cli'], [], global_bear_list, {}, self.queue, console_printer=self.console_printer, debug=True)

class MyBearAction(ResultAction):

    def apply(self, result, original_file_dict, file_diff_dict):
        if False:
            while True:
                i = 10
        logging.debug('ACTION APPLIED SUCCESSFULLY.')

class ProcessingTest_GetDefaultActions(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.section = Section('X')

        def equal(self, other):
            if False:
                print('Hello World!')
            return isinstance(self, other)
        self.eq_apply_patch = ApplyPatchAction.__eq__
        self.eq_print_debug_message = PrintDebugMessageAction.__eq__
        ApplyPatchAction.__eq__ = equal
        PrintDebugMessageAction.__eq__ = equal

    def tearDown(self):
        if False:
            print('Hello World!')
        ApplyPatchAction.__eq__ = self.eq_apply_patch
        PrintDebugMessageAction.__eq__ = self.eq_print_debug_message

    def test_no_key(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(get_default_actions(self.section, []), ({}, {}))

    def test_no_value(self):
        if False:
            for i in range(10):
                print('nop')
        self.section.append(Setting('default_actions', ''))
        self.assertEqual(get_default_actions(self.section, []), ({}, {}))

    def test_only_valid_actions(self):
        if False:
            for i in range(10):
                print('nop')
        self.section.append(Setting('default_actions', 'MyBear: PrintDebugMessageAction, ValidBear: ApplyPatchAction'))
        self.assertEqual(get_default_actions(self.section, []), ({'MyBear': PrintDebugMessageAction, 'ValidBear': ApplyPatchAction}, {}))

    def test_valid_and_invalid_actions(self):
        if False:
            for i in range(10):
                print('nop')
        self.section.append(Setting('default_actions', 'MyBear: INVALID_action, ValidBear: ApplyPatchAction, XBear: ABC'))
        self.assertEqual(get_default_actions(self.section, []), ({'ValidBear': ApplyPatchAction}, {'MyBear': 'INVALID_action', 'XBear': 'ABC'}))

    def test_valid_bear_actions(self):
        if False:
            for i in range(10):
                print('nop')
        my_bear_action = MyBearAction()
        self.section.append(Setting('default_actions', 'MyBear: MyBearAction, ValidBear: ApplyPatchAction'))
        self.assertEqual(get_default_actions(self.section, [my_bear_action]), ({'MyBear': my_bear_action, 'ValidBear': ApplyPatchAction}, {}))

class ProcessingTest_AutoapplyActions(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.log_queue = queue.Queue()
        self.log_printer = ProcessingTestLogPrinter(self.log_queue)
        self.resultX = Result('XBear', 'msg0', actions=[MyBearAction()])
        self.resultY = Result('YBear', 'msg1')
        self.resultZ = Result('ZBear', 'msg2')
        self.results = [self.resultX, self.resultY, self.resultZ]
        self.section = Section('A')

    def test_no_default_actions(self):
        if False:
            while True:
                i = 10
        with LogCapture() as capture:
            ret = autoapply_actions(self.results, {}, {}, self.section, self.log_printer)
        self.assertEqual(ret, self.results)
        capture.check()

    def test_with_invalid_action(self):
        if False:
            print('Hello World!')
        self.section.append(Setting('default_actions', 'XBear: nonSENSE_action'))
        with LogCapture() as capture:
            ret = autoapply_actions(self.results, {}, {}, self.section, self.log_printer)
        self.assertEqual(ret, self.results)
        capture.check(('root', 'WARNING', "Selected default action 'nonSENSE_action' for bear 'XBear' does not exist. Ignoring action."))

    def test_without_default_action_and_unapplicable(self):
        if False:
            return 10
        old_is_applicable = ApplyPatchAction.is_applicable
        ApplyPatchAction.is_applicable = staticmethod(lambda *args: 'The ApplyPatchAction cannot be applied')
        self.section.append(Setting('default_actions', 'NoBear: ApplyPatchAction, YBear: ApplyPatchAction'))
        with LogCapture() as capture:
            ret = autoapply_actions(self.results, {}, {}, self.section, self.log_printer)
        self.assertEqual(ret, self.results)
        capture.check(('root', 'WARNING', 'YBear: The ApplyPatchAction cannot be applied'))
        ApplyPatchAction.is_applicable = staticmethod(old_is_applicable)
        self.section.append(Setting('no_autoapply_warn', True))
        with LogCapture() as capture:
            autoapply_actions(self.results, {}, {}, self.section, self.log_printer)
        capture.check()

    def test_applicable_action(self):
        if False:
            print('Hello World!')
        log_printer = self.log_printer

        class TestAction(ResultAction):

            def apply(self, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                logging.debug('ACTION APPLIED SUCCESSFULLY.')
        ACTIONS.append(TestAction())
        self.section.append(Setting('default_actions', 'Z*: TestAction'))
        with LogCapture() as capture:
            ret = autoapply_actions(self.results, {}, {}, self.section, log_printer)
        self.assertEqual(ret, [self.resultX, self.resultY])
        capture.check(('root', 'DEBUG', 'ACTION APPLIED SUCCESSFULLY.'), ('root', 'INFO', "Applied 'TestAction' on the whole project from 'ZBear'."))
        ACTIONS.pop()

    def test_bear_action_applicable(self):
        if False:
            i = 10
            return i + 15
        MyBearAction.is_applicable = staticmethod(lambda *args: True)
        log_printer = self.log_printer
        self.section.append(Setting('default_actions', 'X*: MyBearAction'))
        with LogCapture() as capture:
            ret = autoapply_actions(self.results, {}, {}, self.section, log_printer)
        self.assertEqual(ret, [self.resultY, self.resultZ])
        capture.check(('root', 'DEBUG', 'ACTION APPLIED SUCCESSFULLY.'), ('root', 'INFO', "Applied 'MyBearAction' on the whole project from 'XBear'."))

    def test_bear_action_unapplicable(self):
        if False:
            i = 10
            return i + 15
        MyBearAction.is_applicable = staticmethod(lambda *args: 'The MyBearAction cannot be applied')
        log_printer = self.log_printer
        self.section.append(Setting('default_actions', 'X*: MyBearAction'))
        with LogCapture() as capture:
            ret = autoapply_actions(self.results, {}, {}, self.section, log_printer)
        self.assertEqual(ret, self.results)
        capture.check(('root', 'WARNING', 'XBear: The MyBearAction cannot be applied'))

    def test_wrong_bear_origin(self):
        if False:
            print('Hello World!')
        MyBearAction.is_applicable = staticmethod(lambda *args: True)
        log_printer = self.log_printer
        self.section.append(Setting('default_actions', 'Y*: MyBearAction'))
        with LogCapture() as capture:
            ret = autoapply_actions(self.results, {}, {}, self.section, log_printer)
        self.assertEqual(ret, self.results)
        capture.check()

    def test_failing_action(self):
        if False:
            while True:
                i = 10

        class FailingTestAction(ResultAction):

            def apply(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                raise RuntimeError("YEAH THAT'S A FAILING BEAR")
        ACTIONS.append(FailingTestAction())
        self.section.append(Setting('default_actions', 'YBear: FailingTestAction'))
        with LogCapture() as capture:
            ret = autoapply_actions(self.results, {}, {}, self.section, self.log_printer)
        self.assertEqual(ret, self.results)
        capture.check(('root', 'ERROR', "Failed to execute action 'FailingTestAction' with error: YEAH THAT'S A FAILING BEAR."), ('root', 'INFO', StringComparison("(?s).*YEAH THAT'S A FAILING BEAR.*")), ('root', 'DEBUG', '-> for result ' + repr(self.resultY) + '.'))
        ACTIONS.pop()

class ProcessingTest_PrintResult(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.section = Section('name')
        self.log_printer = LogPrinter(ConsolePrinter(), log_level=0)
        self.console_printer = ConsolePrinter()

    def test_autoapply_override(self):
        if False:
            return 10
        "\n        Tests that the default_actions aren't automatically applied when the\n        autoapply setting overrides that.\n        "
        self.section.append(Setting('default_actions', 'somebear: PrintDebugMessageAction'))
        results = [5, HiddenResult('origin', []), Result('somebear', 'message', debug_msg='debug')]
        (retval, newres) = print_result(results, {}, 0, lambda *args: None, self.section, self.log_printer, {}, [], console_printer=self.console_printer)
        self.assertEqual(newres, [])