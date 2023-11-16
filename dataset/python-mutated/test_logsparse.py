import functools
import multiprocessing
import os
from .cmdline_tmpl import CmdlineTmpl
file_basic = '\nfrom viztracer import log_sparse\n\n@log_sparse\ndef f():\n    return 1\n\ndef g():\n    return f()\n\nassert g() == 1\n'
file_stack = '\nfrom viztracer import log_sparse\n\ndef h():\n    return 1\n\ndef f():\n    return h()\n\n@log_sparse(stack_depth=2)\ndef g():\n    return f()\n\nassert g() == 1\nassert g() == 1\n'
file_stack_nested = '\nfrom viztracer import log_sparse\n\n@log_sparse(stack_depth=2)\ndef h():\n    return 1\n\ndef f():\n    return h()\n\n@log_sparse(stack_depth=2)\ndef g():\n    return f()\n\nassert g() == 1\nassert g() == 1\n'
file_multiprocess = '\nfrom multiprocessing import Process\nfrom viztracer import log_sparse\nimport time\n\n@log_sparse\ndef f(x):\n    return x*x\n\nif __name__ == "__main__":\n    for i in range(3):\n        p = Process(target=f, args=(i,))\n        p.start()\n        p.join()\n        time.sleep(0.1)\n'
file_context_manager = '\nfrom viztracer import VizTracer, log_sparse\n\n@log_sparse(dynamic_tracer_check=True)\ndef f():\n    return 1\n\ndef g():\n    return f()\n\n@log_sparse\ndef h():\n    return 2\n\n@log_sparse(dynamic_tracer_check=True, stack_depth=1)\ndef q():\n    return 3\n\nif __name__ == "__main__":\n    with VizTracer(output_file="result.json"):\n        assert g() == 1\n        assert h() == 2\n        assert q() == 3\n'
file_context_manager_logsparse = '\nfrom viztracer import VizTracer, log_sparse\n\n@log_sparse(dynamic_tracer_check=True)\ndef f():\n    return 1\n\ndef g():\n    return f()\n\n@log_sparse\ndef h():\n    return 2\n\n@log_sparse(dynamic_tracer_check=True, stack_depth=1)\ndef q():\n    return 3\n\nif __name__ == "__main__":\n    with VizTracer(output_file="result.json", log_sparse=True):\n        assert g() == 1\n        assert h() == 2\n        assert q() == 3\n'
file_context_manager_logsparse_stack = '\nfrom viztracer import VizTracer, log_sparse\n\n@log_sparse(dynamic_tracer_check=True)\ndef f():\n    return 1\n\n@log_sparse(dynamic_tracer_check=True, stack_depth=1)\ndef g():\n    return f()\n\n@log_sparse(dynamic_tracer_check=True)\ndef h():\n    return 2\n\nif __name__ == "__main__":\n    assert g() == 1\n    assert h() == 2\n\n    with VizTracer(output_file="result.json", log_sparse=True):\n        assert g() == 1\n        assert h() == 2\n'

class TestLogSparse(CmdlineTmpl):

    def check_func(self, data, target):
        if False:
            return 10
        names = [entry['name'] for entry in data['traceEvents']]
        function_names = [name.split(' ')[0] for name in names if name not in ['process_name', 'thread_name']]
        self.assertEqual(function_names, target)

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        self.template(['viztracer', '-o', 'result.json', '--log_sparse', 'cmdline_test.py'], script=file_basic, expected_output_file='result.json', expected_entries=1, check_func=functools.partial(self.check_func, target=['f']))
        self.template(['viztracer', '-o', 'result.json', 'cmdline_test.py'], script=file_basic, expected_output_file='result.json')

    def test_stack(self):
        if False:
            while True:
                i = 10
        self.template(['viztracer', '-o', 'result.json', '--log_sparse', 'cmdline_test.py'], script=file_stack, expected_output_file='result.json', expected_entries=4, check_func=functools.partial(self.check_func, target=['f', 'g', 'f', 'g']))
        self.template(['viztracer', '-o', 'result.json', '--log_sparse', 'cmdline_test.py'], script=file_stack_nested, expected_output_file='result.json', expected_entries=4, check_func=functools.partial(self.check_func, target=['f', 'g', 'f', 'g']))

    def test_without_tracer(self):
        if False:
            print('Hello World!')
        self.template(['python', 'cmdline_test.py'], script=file_basic, expected_output_file=None)
        self.template(['python', 'cmdline_test.py'], script=file_stack, expected_output_file=None)

    def test_multiprocess(self):
        if False:
            for i in range(10):
                print('nop')
        if multiprocessing.get_start_method() == 'fork':
            try:
                self.template(['viztracer', '-o', 'result.json', '--log_sparse', 'cmdline_test.py'], script=file_multiprocess, expected_output_file='result.json', expected_entries=3, check_func=functools.partial(self.check_func, target=['f', 'f', 'f']), concurrency='multiprocessing')
            except Exception as e:
                if not os.getenv('COVERAGE_RUN'):
                    raise e

    def test_context_manager(self):
        if False:
            for i in range(10):
                print('nop')
        self.template(['python', 'cmdline_test.py'], script=file_context_manager, expected_output_file='result.json', expected_entries=4, check_func=functools.partial(self.check_func, target=['f', 'g', 'h', 'q']))
        self.template(['python', 'cmdline_test.py'], script=file_context_manager_logsparse, expected_output_file='result.json', expected_entries=2, check_func=functools.partial(self.check_func, target=['f', 'q']))
        self.template(['python', 'cmdline_test.py'], script=file_context_manager_logsparse_stack, expected_output_file='result.json', expected_entries=2, check_func=functools.partial(self.check_func, target=['g', 'h']))