import json
import logging
import lzma
import os
import tempfile
import unittest
import zlib
from collections import namedtuple
from functools import wraps
from shutil import copyfileobj
from typing import Callable, List, Optional, Tuple, overload
from .cmdline_tmpl import CmdlineTmpl
from .test_performance import Timer
from .util import get_tests_data_file_path

class TestVCompressor(CmdlineTmpl):

    def test_basic(self):
        if False:
            while True:
                i = 10
        with tempfile.TemporaryDirectory() as tmpdir:
            cvf_path = os.path.join(tmpdir, 'result.cvf')
            dup_json_path = os.path.join(tmpdir, 'result.json')
            self.template(['viztracer', '-o', cvf_path, '--compress', get_tests_data_file_path('multithread.json')], expected_output_file=cvf_path, cleanup=False)
            self.template(['viztracer', '-o', dup_json_path, '--decompress', cvf_path], expected_output_file=dup_json_path)

    def test_compress_invalid(self):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as tmpdir:
            cvf_path = os.path.join(tmpdir, 'result.cvf')
            not_exist_path = os.path.join(tmpdir, 'do_not_exist.json')
            result = self.template(['viztracer', '-o', cvf_path, '--compress', not_exist_path], expected_output_file=None, success=False)
            self.assertIn('Unable to find file', result.stdout.decode('utf8'))
            result = self.template(['viztracer', '-o', cvf_path, '--compress', get_tests_data_file_path('fib.py')], expected_output_file=None, success=False)
            self.assertIn('Only support compressing json report', result.stdout.decode('utf8'))

    def test_compress_default_outputfile(self):
        if False:
            while True:
                i = 10
        default_compress_output = 'result.cvf'
        self.template(['viztracer', '--compress', get_tests_data_file_path('multithread.json')], expected_output_file=default_compress_output, cleanup=False)
        self.assertTrue(os.path.exists(default_compress_output))
        self.template(['viztracer', '-o', 'result.json', '--decompress', default_compress_output], expected_output_file='result.json')
        self.cleanup(output_file=default_compress_output)

    def test_decompress_invalid(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmpdir:
            not_exist_path = os.path.join(tmpdir, 'result.cvf')
            dup_json_path = os.path.join(tmpdir, 'result.json')
            result = self.template(['viztracer', '-o', dup_json_path, '--decompress', not_exist_path], expected_output_file=dup_json_path, success=False)
            self.assertIn('Unable to find file', result.stdout.decode('utf8'))

    def test_decompress_default_outputfile(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmpdir:
            cvf_path = os.path.join(tmpdir, 'result.cvf')
            default_decompress_output = 'result.json'
            self.template(['viztracer', '-o', cvf_path, '--compress', get_tests_data_file_path('multithread.json')], expected_output_file=cvf_path, cleanup=False)
            self.template(['viztracer', '--decompress', cvf_path], expected_output_file=default_decompress_output, cleanup=False)
            self.assertTrue(os.path.exists(default_decompress_output))
            self.cleanup(output_file=default_decompress_output)
test_large_fib = "\nfrom viztracer import VizTracer\ntracer = VizTracer(tracer_entries=2000000)\ntracer.start()\n\ndef fib(n):\n    if n < 2:\n        return 1\n    return fib(n-1) + fib(n-2)\nfib(27)\n\ntracer.stop()\ntracer.save(output_file='%s')\n"

class TestVCompressorPerformance(CmdlineTmpl):
    BenchmarkResult = namedtuple('BenchmarkResult', ['file_size', 'elapsed_time'])

    @overload
    def _benchmark(benchmark_process: Callable[..., None]):
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def _benchmark(repeat: int):
        if False:
            print('Hello World!')
        ...

    def _benchmark(*args, **kargs):
        if False:
            while True:
                i = 10

        def _decorator(benchmark_process: Callable) -> Callable:
            if False:
                while True:
                    i = 10

            @wraps(benchmark_process)
            def _wrapper(self, uncompressed_file_path: str) -> 'TestVCompressorPerformance.BenchmarkResult':
                if False:
                    return 10
                compression_time_total = 0.0
                with tempfile.TemporaryDirectory() as tmpdir:
                    compressed_file_path = os.path.join(tmpdir, 'result.compressed')
                    benchmark_process(self, uncompressed_file_path, compressed_file_path)
                    os.remove(compressed_file_path)
                    for _ in range(loop_time):
                        with Timer() as t:
                            benchmark_process(self, uncompressed_file_path, compressed_file_path)
                            compression_time_total += t.get_time()
                        compressed_file_size = os.path.getsize(compressed_file_path)
                        os.remove(compressed_file_path)
                return TestVCompressorPerformance.BenchmarkResult(compressed_file_size, compression_time_total / loop_time)
            return _wrapper
        if len(args) == 0 and len(kargs) == 0:
            raise TypeError('_benchmark must decorate a function.')
        if len(args) == 1 and len(kargs) == 0 and callable(args[0]):
            loop_time = 3
            return _decorator(args[0])
        loop_time = kargs['repeat'] if 'repeat' in kargs else args[0]
        return _decorator

    @staticmethod
    def _human_readable_filesize(filesize: int) -> str:
        if False:
            return 10
        units = [('PB', 1 << 50), ('TB', 1 << 40), ('GB', 1 << 30), ('MB', 1 << 20), ('KB', 1 << 10)]
        for (unit_name, unit_base) in units:
            norm_size = filesize / unit_base
            if norm_size >= 0.8:
                return f'{norm_size:8.2f}{unit_name}'
        return f'{filesize:8.2f}B'

    @classmethod
    def _print_result(cls, filename: str, original_size: int, vcompress_result: BenchmarkResult, other_results: List[Tuple[str, BenchmarkResult]], subtest_idx: Optional[int]=None):
        if False:
            for i in range(10):
                print('nop')
        if subtest_idx is None:
            logging.info(f'On file "{filename}":')
        else:
            logging.info(f'{subtest_idx}. On file "{filename}":')
        logging.info('    [Space]')
        logging.info('      Uncompressed:   {}'.format(cls._human_readable_filesize(original_size)))
        logging.info('      VCompressor:    {}(1.000) [CR:{:6.2f}%]'.format(cls._human_readable_filesize(vcompress_result.file_size), vcompress_result.file_size / original_size * 100))
        for (name, result) in other_results:
            logging.info('      {}{}({:.3f}) [CR:{:6.2f}%]'.format(name + ':' + ' ' * max(15 - len(name), 0), cls._human_readable_filesize(result.file_size), result.file_size / vcompress_result.file_size, result.file_size / original_size * 100))
        logging.info('    [Time]')
        logging.info('      VCompressor:    {:9.3f}s(1.000)'.format(vcompress_result.elapsed_time))
        for (name, result) in other_results:
            logging.info('      {}{:9.3f}s({:.3f})'.format(name + ':' + ' ' * max(15 - len(name), 0), result.elapsed_time, result.elapsed_time / vcompress_result.elapsed_time))

    @_benchmark
    def _benchmark_vcompressor(self, uncompressed_file_path: str, compressed_file_path: str) -> None:
        if False:
            while True:
                i = 10
        self.template(['viztracer', '-o', compressed_file_path, '--compress', uncompressed_file_path], expected_output_file=compressed_file_path, script=None, cleanup=False)

    @_benchmark
    def _benchmark_lzma(self, uncompressed_file_path: str, compressed_file_path: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        with open(uncompressed_file_path, 'rb') as original_file:
            with lzma.open(compressed_file_path, 'wb', preset=lzma.PRESET_DEFAULT) as compressed_file:
                copyfileobj(original_file, compressed_file)

    @_benchmark
    def _benchmark_zlib(self, uncompressed_file_path: str, compressed_file_path: str) -> None:
        if False:
            i = 10
            return i + 15
        with open(uncompressed_file_path, 'rb') as original_file:
            compressed_data = zlib.compress(original_file.read())
        with open(compressed_file_path, 'wb') as compressed_file:
            compressed_file.write(compressed_data)

    @_benchmark
    def _benchmark_vcompressor_lzma(self, uncompressed_file_path: str, compressed_file_path: str) -> None:
        if False:
            i = 10
            return i + 15
        tmp_compress_file = uncompressed_file_path + '.tmp'
        self.template(['viztracer', '-o', tmp_compress_file, '--compress', uncompressed_file_path], expected_output_file=tmp_compress_file, script=None, cleanup=False)
        with open(tmp_compress_file, 'rb') as tmp_file:
            with lzma.open(compressed_file_path, 'wb', preset=lzma.PRESET_DEFAULT) as compressed_file:
                copyfileobj(tmp_file, compressed_file)

    @_benchmark
    def _benchmark_vcompressor_zlib(self, uncompressed_file_path: str, compressed_file_path: str) -> None:
        if False:
            return 10
        tmp_compress_file = uncompressed_file_path + '.tmp'
        self.template(['viztracer', '-o', tmp_compress_file, '--compress', uncompressed_file_path], expected_output_file=tmp_compress_file, script=None, cleanup=False)
        with open(tmp_compress_file, 'rb') as tmp_file:
            compressed_data = zlib.compress(tmp_file.read())
        with open(compressed_file_path, 'wb') as compressed_file:
            compressed_file.write(compressed_data)

    def test_benchmark_basic(self):
        if False:
            i = 10
            return i + 15
        testcases_filename = ['vdb_basic.json', 'multithread.json']
        for (subtest_idx, filename) in enumerate(testcases_filename, start=1):
            path = get_tests_data_file_path(filename)
            original_size = os.path.getsize(path)
            other_results = [('LZMA', self._benchmark_lzma(path))]
            with self.subTest(testcase=filename):
                vcompress_result = self._benchmark_vcompressor(path)
                self._print_result(filename, original_size, vcompress_result, other_results, subtest_idx=subtest_idx)

    @unittest.skipUnless(os.getenv('GITHUB_ACTIONS'), 'skipped because not in github actions')
    def test_benchmark_large_file(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmpdir:
            origin_json_path = os.path.join(tmpdir, 'large_fib.json')
            run_script = test_large_fib % origin_json_path.replace('\\', '/')
            self.template(['python', 'cmdline_test.py'], script=run_script, cleanup=False, expected_output_file=origin_json_path)
            original_size = os.path.getsize(origin_json_path)
            other_results = [('LZMA', self._benchmark_lzma(origin_json_path)), ('ZLIB', self._benchmark_zlib(origin_json_path)), ('VC+LZMA', self._benchmark_vcompressor_lzma(origin_json_path)), ('VC+ZLIB', self._benchmark_vcompressor_zlib(origin_json_path))]
            with self.subTest(testcase='large_fib.json'):
                vcompress_result = self._benchmark_vcompressor(origin_json_path)
                self._print_result('large_fib.json', original_size, vcompress_result, other_results)

class VCompressorCompare(unittest.TestCase):

    def assertEventsEqual(self, first: list, second: list, ts_margin: float):
        if False:
            return 10
        '\n        This method is used to assert if two lists of events are equal,\n        first and second are the two lists that we compare,\n        ts_margin is the max timestamps diff that we tolerate.\n        The timestamps may changed before/after the compression for more effective compression\n        '
        self.assertEqual(len(first), len(second), f'list length not equal, first is {len(first)} \n second is {len(second)}')
        first.sort(key=lambda i: i['ts'])
        second.sort(key=lambda i: i['ts'])
        for i in range(len(first)):
            self.assertEventEqual(first[i], second[i], ts_margin)

    def assertEventEqual(self, first: dict, second: dict, ts_margin: float):
        if False:
            print('Hello World!')
        '\n        This method is used to assert if two events are equal,\n        first and second are the two events that we compare,\n        ts_margin is the max timestamps diff that we tolerate.\n        The timestamps may changed before/after the compression for more effective compression\n        '
        self.assertEqual(len(first), len(second), f'event length not equal, first is: \n {str(first)} \n second is: \n {str(second)}')
        for (key, value) in first.items():
            if key in ['ts', 'dur']:
                self.assertGreaterEqual(ts_margin, abs(value - second[key]), f'{key} diff is greater than margin')
            else:
                self.assertEqual(value, second[key], f'{key} is not equal')

    def assertThreadOrProcessEqual(self, first: list, second: list):
        if False:
            while True:
                i = 10
        '\n        This method is used to assert if two lists of thread names are equal\n        '
        self.assertEqual(len(first), len(second), f'list length not equal, first is {len(first)} \n second is {len(second)}')
        first.sort(key=lambda i: (i['pid'], i['tid']))
        second.sort(key=lambda i: (i['pid'], i['tid']))
        for _ in range(len(first)):
            self.assertEqual(first, second, f'{first} and {second} not equal')
test_counter_events = '\nimport threading\nimport time\nimport sys\nfrom viztracer import VizTracer, VizCounter\n\ntracer = VizTracer()\ntracer.start()\n\nclass MyThreadSparse(threading.Thread):\n    def run(self):\n        counter = VizCounter(tracer, \'thread counter \' + str(self.ident))\n        counter.a = sys.maxsize - 1\n        time.sleep(0.01)\n        counter.a = sys.maxsize * 2\n        time.sleep(0.01)\n        counter.a = -sys.maxsize + 2\n        time.sleep(0.01)\n        counter.a = -sys.maxsize * 2\n\nmain_counter = VizCounter(tracer, \'main counter\')\nthread1 = MyThreadSparse()\nthread2 = MyThreadSparse()\nmain_counter.arg1 = 100.01\nmain_counter.arg2 = -100.01\nmain_counter.arg3 = 0.0\ndelattr(main_counter, "arg3")\n\nthread1.start()\nthread2.start()\n\nthreads = [thread1, thread2]\n\nfor thread in threads:\n    thread.join()\n\nmain_counter.arg1 = 200.01\nmain_counter.arg2 = -200.01\n\ntracer.stop()\ntracer.save(output_file=\'%s\')\n'
test_duplicated_timestamp = "\nfrom viztracer import VizTracer\ntracer = VizTracer(tracer_entries=1000000)\ntracer.start()\n\ndef call_self(n):\n    if n == 0:\n        return\n    return call_self(n-1)\nfor _ in range(10):\n    call_self(1000)\n\ntracer.stop()\ntracer.save(output_file='%s')\n"
test_non_frequent_events = '\nimport threading\nfrom viztracer import VizTracer, VizObject\n\ntracer = VizTracer()\ntracer.start()\n\nclass MyThreadSparse(threading.Thread):\n    def run(self):\n        viz_object = VizObject(tracer, \'thread object \' + str(self.ident))\n        viz_object.a = \'test string 1\'\n        viz_object.a = \'test string 2\'\n        viz_object.a = {\'test\': \'string3\'}\n        viz_object.a = [\'test string 4\']\n        tracer.log_instant("thread id " + str(self.ident))\n        tracer.log_instant("thread id " + str(self.ident), "test instant string", "t")\n        tracer.log_instant("thread id " + str(self.ident), {"b":"test"}, "g")\n        tracer.log_instant("thread id " + str(self.ident), {"b":"test", "c":123}, "p")\n\nmain_viz_object = VizObject(tracer, \'main viz_object\')\nthread1 = MyThreadSparse()\nthread2 = MyThreadSparse()\nmain_viz_object.arg1 = 100.01\nmain_viz_object.arg2 = -100.01\nmain_viz_object.arg3 = [100, -100]\ndelattr(main_viz_object, \'arg3\')\ntracer.log_instant("process")\ntracer.log_instant("process", "test instant string", "t")\ntracer.log_instant("process", {"b":"test"}, "g")\ntracer.log_instant("process", {"b":"test", "c":123}, "p")\n\nthread1.start()\nthread2.start()\nthreads = [thread1, thread2]\n\nfor thread in threads:\n    thread.join()\n\nmain_viz_object.arg1 = {100: "string1"}\nmain_viz_object.arg2 = {100: "string1", -100: "string2"}\n\ntracer.stop()\ntracer.save(output_file=\'%s\')\n'
test_fee_args = "\nfrom viztracer import VizTracer\ntracer = VizTracer(log_func_args=True, log_func_retval=True)\ntracer.start()\n\ndef fib(n):\n    if n < 2:\n        return 1\n    return fib(n-1) + fib(n-2)\nfib(10)\ntracer.log_func_args = False\ntracer.log_func_retval = False\nfib(10)\n\ntracer.stop()\ntracer.save(output_file='%s')\n"

class TestVCompressorCorrectness(CmdlineTmpl, VCompressorCompare):

    def _generate_test_data(self, test_file):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.TemporaryDirectory() as tmpdir:
            cvf_path = os.path.join(tmpdir, 'result.cvf')
            dup_json_path = os.path.join(tmpdir, 'result.json')
            origin_json_path = get_tests_data_file_path(test_file)
            self.template(['viztracer', '-o', cvf_path, '--compress', origin_json_path], expected_output_file=cvf_path, cleanup=False)
            self.template(['viztracer', '-o', dup_json_path, '--decompress', cvf_path], expected_output_file=dup_json_path, cleanup=False)
            with open(origin_json_path, 'r') as f:
                origin_json_data = json.load(f)
            with open(dup_json_path, 'r') as f:
                dup_json_data = json.load(f)
        return (origin_json_data, dup_json_data)

    def _generate_test_data_by_script(self, run_script):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as tmpdir:
            origin_json_path = os.path.join(tmpdir, 'result.json')
            cvf_path = os.path.join(tmpdir, 'result.cvf')
            dup_json_path = os.path.join(tmpdir, 'recovery.json')
            run_script = run_script % origin_json_path.replace('\\', '/')
            self.template(['python', 'cmdline_test.py'], script=run_script, cleanup=False, expected_output_file=origin_json_path)
            self.template(['viztracer', '-o', cvf_path, '--compress', origin_json_path], expected_output_file=cvf_path, cleanup=False)
            self.template(['viztracer', '-o', dup_json_path, '--decompress', cvf_path], expected_output_file=dup_json_path, cleanup=False)
            with open(origin_json_path, 'r') as f:
                origin_json_data = json.load(f)
            with open(dup_json_path, 'r') as f:
                dup_json_data = json.load(f)
        return (origin_json_data, dup_json_data)

    def test_file_info(self):
        if False:
            i = 10
            return i + 15
        (origin_json_data, dup_json_data) = self._generate_test_data('multithread.json')
        self.assertEqual(origin_json_data['file_info'], dup_json_data['file_info'])

    def test_process_name(self):
        if False:
            i = 10
            return i + 15
        (origin_json_data, dup_json_data) = self._generate_test_data('multithread.json')
        origin_names = [i for i in origin_json_data['traceEvents'] if i['ph'] == 'M' and i['name'] == 'process_name']
        dup_names = [i for i in dup_json_data['traceEvents'] if i['ph'] == 'M' and i['name'] == 'process_name']
        self.assertThreadOrProcessEqual(origin_names, dup_names)

    def test_thread_name(self):
        if False:
            return 10
        (origin_json_data, dup_json_data) = self._generate_test_data('multithread.json')
        origin_names = [i for i in origin_json_data['traceEvents'] if i['ph'] == 'M' and i['name'] == 'thread_name']
        dup_names = [i for i in dup_json_data['traceEvents'] if i['ph'] == 'M' and i['name'] == 'thread_name']
        self.assertThreadOrProcessEqual(origin_names, dup_names)

    def test_fee(self):
        if False:
            for i in range(10):
                print('nop')
        (origin_json_data, dup_json_data) = self._generate_test_data('multithread.json')
        origin_fee_events = {}
        for event in origin_json_data['traceEvents']:
            if event['ph'] == 'X':
                event_key = (event['pid'], event['tid'])
                if event_key not in origin_fee_events:
                    origin_fee_events[event_key] = []
                origin_fee_events[event_key].append(event)
        dup_fee_events = {}
        for event in dup_json_data['traceEvents']:
            if event['ph'] == 'X':
                event_key = (event['pid'], event['tid'])
                if event_key not in dup_fee_events:
                    self.assertIn(event_key, origin_fee_events, f'thread data {str(event_key)} not in origin data')
                    dup_fee_events[event_key] = []
                dup_fee_events[event_key].append(event)
        for (key, value) in origin_fee_events.items():
            self.assertIn(key, dup_fee_events, f'thread data {str(key)} not in decompressed data')
            self.assertEventsEqual(value, dup_fee_events[key], 0.011)

    def test_fee_with_args(self):
        if False:
            print('Hello World!')
        (origin_json_data, dup_json_data) = self._generate_test_data_by_script(test_fee_args)
        origin_fee_events = [i for i in origin_json_data['traceEvents'] if i['ph'] == 'X']
        dup_fee_events = [i for i in dup_json_data['traceEvents'] if i['ph'] == 'X']
        self.assertEventsEqual(origin_fee_events, dup_fee_events, 0.011)

    def test_counter_events(self):
        if False:
            return 10
        (origin_json_data, dup_json_data) = self._generate_test_data_by_script(test_counter_events)
        origin_counter_events = [i for i in origin_json_data['traceEvents'] if i['ph'] == 'C']
        dup_counter_events = [i for i in dup_json_data['traceEvents'] if i['ph'] == 'C']
        self.assertEventsEqual(origin_counter_events, dup_counter_events, 0.011)

    def test_duplicated_timestamp(self):
        if False:
            for i in range(10):
                print('nop')
        (origin_json_data, dup_json_data) = self._generate_test_data_by_script(test_duplicated_timestamp)
        origin_fee_events = [i for i in origin_json_data['traceEvents'] if i['ph'] == 'X']
        dup_fee_events = [i for i in dup_json_data['traceEvents'] if i['ph'] == 'X']
        dup_timestamp_list = [event['ts'] for event in dup_fee_events if event['ph'] == 'X']
        dup_timestamp_set = set(dup_timestamp_list)
        self.assertEqual(len(dup_timestamp_list), len(dup_timestamp_set), "There's duplicated timestamp")
        self.assertEventsEqual(origin_fee_events, dup_fee_events, 0.011)

    def test_non_frequent_events(self):
        if False:
            for i in range(10):
                print('nop')
        (origin_json_data, dup_json_data) = self._generate_test_data_by_script(test_non_frequent_events)
        ph_filter = ['X', 'M', 'C']
        origin_events = [i for i in origin_json_data['traceEvents'] if i['ph'] not in ph_filter]
        dup_events = [i for i in dup_json_data['traceEvents'] if i['ph'] not in ph_filter]
        self.assertEventsEqual(origin_events, dup_events, 0.011)