"""
please make sure to run in the tools path
usage: python sampcd_processor.py --mode {cpu or gpu}
    {cpu or gpu}: running in cpu version or gpu version

for example, you can run cpu version testing like this:

    python sampcd_processor.py --mode cpu

"""
import collections
import functools
import multiprocessing
import os
import platform
import queue
import re
import threading
import time
import typing
import xdoctest
from sampcd_processor_utils import TEST_TIMEOUT, DocTester, TestResult, log_exit, logger, parse_args, run_doctest
XDOCTEST_CONFIG = {'global_exec': '\\n'.join(['import paddle', "paddle.device.set_device('cpu')", "paddle.set_default_dtype('float32')", 'paddle.disable_static()']), 'default_runtime_state': {'IGNORE_WHITESPACE': True}}

def _patch_global_state(debug, verbose):
    if False:
        while True:
            i = 10
    from xdoctest import global_state
    _debug_xdoctest = debug and verbose > 2
    global_state.DEBUG = _debug_xdoctest
    global_state.DEBUG_PARSER = global_state.DEBUG_PARSER and _debug_xdoctest
    global_state.DEBUG_CORE = global_state.DEBUG_CORE and _debug_xdoctest
    global_state.DEBUG_RUNNER = global_state.DEBUG_RUNNER and _debug_xdoctest
    global_state.DEBUG_DOCTEST = global_state.DEBUG_DOCTEST and _debug_xdoctest

def _patch_tensor_place():
    if False:
        print('Hello World!')
    from xdoctest import checker
    pattern_tensor = re.compile('\n        (Tensor\\(.*?place=)     # Tensor start\n        (.*?)                   # Place=(XXX)\n        (\\,.*?\\))\n        ', re.X | re.S)
    _check_output = checker.check_output

    def check_output(got, want, runstate=None):
        if False:
            print('Hello World!')
        if not want:
            return True
        return _check_output(got=pattern_tensor.sub('\\1Place(cpu)\\3', got), want=pattern_tensor.sub('\\1Place(cpu)\\3', want), runstate=runstate)
    checker.check_output = check_output

def _patch_float_precision(digits):
    if False:
        while True:
            i = 10
    from xdoctest import checker
    pattern_number = re.compile('\n        (?:\n            (?:(?<=[\\s*\\[\\(\\\'\\"\\:])|^)                  # number starts\n            (?:                                         # int/float or complex-real\n                (?:\n                    [+-]?\n                    (?:\n                        (?: \\d*\\.\\d+) | (?: \\d+\\.?)     # int/float\n                    )\n                )\n                (?:[Ee][+-]?\\d+)?\n            )\n            (?:                                         # complex-imag\n                (?:\n                    (?:\n                        [+-]?\n                        (?:\n                            (?: \\d*\\.\\d+) | (?: \\d+\\.?)\n                        )\n                    )\n                    (?:[Ee][+-]?\\d+)?\n                )\n            (?:[Jj])\n            )?\n        )\n        ', re.X | re.S)
    _check_output = checker.check_output

    def _sub_number(match_obj, digits):
        if False:
            for i in range(10):
                print('nop')
        match_str = match_obj.group()
        if 'j' in match_str or 'J' in match_str:
            try:
                match_num = complex(match_str)
            except ValueError:
                return match_str
            return str(complex(round(match_num.real, digits), round(match_num.imag, digits))).strip('(').strip(')')
        else:
            try:
                return str(round(float(match_str), digits))
            except ValueError:
                return match_str
    sub_number = functools.partial(_sub_number, digits=digits)

    def check_output(got, want, runstate=None):
        if False:
            i = 10
            return i + 15
        if not want:
            return True
        return _check_output(got=pattern_number.sub(sub_number, got), want=pattern_number.sub(sub_number, want), runstate=runstate)
    checker.check_output = check_output

class Directive:
    """Base class of global direvtives just for `xdoctest`."""
    pattern: typing.Pattern

    def parse_directive(self, docstring: str) -> typing.Tuple[str, typing.Any]:
        if False:
            print('Hello World!')
        pass

class TimeoutDirective(Directive):
    pattern = re.compile('\n        (?:\n            (?:\n                \\s*\\>{3}\\s*\\#\\s*x?doctest\\:\\s*\n            )\n            (?P<op>[\\+\\-])\n            (?:\n                TIMEOUT\n            )\n            \\(\n                (?P<time>\\d+)\n            \\)\n            (?:\n                \\s*?\n            )\n        )\n        ', re.X | re.S)

    def __init__(self, timeout):
        if False:
            for i in range(10):
                print('nop')
        self._timeout = timeout

    def parse_directive(self, docstring):
        if False:
            return 10
        match_obj = self.pattern.search(docstring)
        if match_obj is not None:
            op_time = match_obj.group('time')
            match_start = match_obj.start()
            match_end = match_obj.end()
            return (docstring[:match_start] + '\n' + docstring[match_end:], float(op_time))
        return (docstring, float(self._timeout))

class SingleProcessDirective(Directive):
    pattern = re.compile('\n        (?:\n            (?:\n                \\s*\\>{3}\\s*\\#\\s*x?doctest\\:\\s*\n            )\n            (?P<op>[\\+\\-])\n            (?:\n                SOLO\n            )\n            (?:\n                (?P<reason>.*?)\n            )\n            \\s\n        )\n        ', re.X | re.S)

    def parse_directive(self, docstring):
        if False:
            print('Hello World!')
        match_obj = self.pattern.search(docstring)
        if match_obj is not None:
            op_reason = match_obj.group('reason')
            match_start = match_obj.start()
            match_end = match_obj.end()
            return (docstring[:match_start] + '\n' + docstring[match_end:], op_reason)
        return (docstring, None)

class BadStatement:
    msg: str = ''

    def check(self, docstring: str) -> bool:
        if False:
            print('Hello World!')
        'Return `True` for bad statement detected.'
        raise NotImplementedError

class Fluid(BadStatement):
    msg = 'Please do NOT use `fluid` api.'
    _pattern = re.compile('\n        (\\>{3}|\\.{3})\n        (?P<comment>.*)\n        import\n        .*\n        (\\bfluid\\b)\n        ', re.X)

    def check(self, docstring):
        if False:
            for i in range(10):
                print('nop')
        for match_obj in self._pattern.finditer(docstring):
            comment = match_obj.group('comment').strip()
            if not comment.startswith('#'):
                return True
        return False

class SkipNoReason(BadStatement):
    msg = 'Please add sample code skip reason.'
    _pattern = re.compile('\n        \\#\n        \\s*\n        (x?doctest:)\n        \\s*\n        [+]SKIP\n        (?P<reason>.*)\n        ', re.X)

    def check(self, docstring):
        if False:
            print('Hello World!')
        for match_obj in self._pattern.finditer(docstring):
            reason = match_obj.group('reason').strip().strip('(').strip(')').strip()
            if not reason:
                return True
        return False

class DeprecatedRequired(BadStatement):
    msg = 'Please use `# doctest: +REQUIRES({})` instead of `# {} {}`.'
    _pattern = re.compile('\n        \\#\n        \\s*\n        (?P<directive>require[sd]?\\s*:)\n        (?P<env>.+)\n        ', re.X)

    def check(self, docstring):
        if False:
            i = 10
            return i + 15
        for match_obj in self._pattern.finditer(docstring):
            dep_directive = match_obj.group('directive').strip()
            dep_env = match_obj.group('env').strip()
            if dep_env:
                env = 'env:' + ', env:'.join([e.strip().upper() for e in dep_env.split(',') if e.strip()])
                self.msg = self.__class__.msg.format(env, dep_directive, dep_env)
                return True
        return False

class Xdoctester(DocTester):
    """A Xdoctest doctester."""
    directives: typing.Dict[str, typing.Tuple[typing.Type[Directive], ...]] = {'timeout': (TimeoutDirective, TEST_TIMEOUT), 'solo': (SingleProcessDirective,)}
    bad_statements: typing.Dict[str, typing.Tuple[typing.Type[BadStatement], ...]] = {'fluid': (Fluid,), 'skip': (SkipNoReason,), 'require': (DeprecatedRequired,)}

    def __init__(self, debug=False, style='freeform', target='codeblock', mode='native', verbose=2, patch_global_state=True, patch_tensor_place=True, patch_float_precision=5, use_multiprocessing=True, **config):
        if False:
            for i in range(10):
                print('nop')
        self.debug = debug
        self.style = style
        self.target = target
        self.mode = mode
        self.verbose = verbose
        self.config = {**XDOCTEST_CONFIG, **(config or {})}
        self._test_capacity = set()
        self._patch_global_state = patch_global_state
        self._patch_tensor_place = patch_tensor_place
        self._patch_float_precision = patch_float_precision
        self._use_multiprocessing = use_multiprocessing
        self._patch_xdoctest()
        self.docstring_parser = functools.partial(xdoctest.core.parse_docstr_examples, style=self.style)
        self.directive_pattern = re.compile('\n            (?<=(\\#\\s))     # positive lookbehind, directive begins\n            (doctest)       # directive prefix, which should be replaced\n            (?=(:\\s*.*\\n))  # positive lookahead, directive content\n            ', re.X)
        self.directive_prefix = 'xdoctest'

    def _patch_xdoctest(self):
        if False:
            print('Hello World!')
        if self._patch_global_state:
            _patch_global_state(self.debug, self.verbose)
        if self._patch_tensor_place:
            _patch_tensor_place()
        if self._patch_float_precision is not None:
            _patch_float_precision(self._patch_float_precision)

    def _parse_directive(self, docstring: str) -> typing.Tuple[str, typing.Dict[str, Directive]]:
        if False:
            print('Hello World!')
        directives = {}
        for (name, directive_cls) in self.directives.items():
            (docstring, direct) = directive_cls[0](*directive_cls[1:]).parse_directive(docstring)
            directives[name] = direct
        return (docstring, directives)

    def convert_directive(self, docstring: str) -> str:
        if False:
            while True:
                i = 10
        'Replace directive prefix with xdoctest'
        return self.directive_pattern.sub(self.directive_prefix, docstring)

    def prepare(self, test_capacity: set):
        if False:
            while True:
                i = 10
        'Set environs for xdoctest directive.\n        The keys in environs, which also used in `# xdoctest: +REQUIRES(env:XX)`, should be UPPER case.\n\n        If `test_capacity = {"cpu"}`, then we set:\n\n            - `os.environ["CPU"] = "True"`\n\n        which makes this SKIPPED:\n\n            - # xdoctest: +REQUIRES(env:GPU)\n\n        If `test_capacity = {"cpu", "gpu"}`, then we set:\n\n            - `os.environ["CPU"] = "True"`\n            - `os.environ["GPU"] = "True"`\n\n        which makes this SUCCESS:\n\n            - # xdoctest: +REQUIRES(env:GPU)\n        '
        logger.info('Set xdoctest environ ...')
        for capacity in test_capacity:
            key = capacity.upper()
            os.environ[key] = 'True'
            logger.info('Environ: %s , set to True.', key)
        logger.info('API check using Xdoctest prepared!-- Example Code')
        logger.info('running under python %s', platform.python_version())
        logger.info('running under xdoctest %s', xdoctest.__version__)
        self._test_capacity = test_capacity

    def _check_bad_statements(self, docstring: str) -> typing.Set[BadStatement]:
        if False:
            i = 10
            return i + 15
        bad_results = set()
        for (_, statement_cls) in self.bad_statements.items():
            bad_statement = statement_cls[0](*statement_cls[1:])
            if bad_statement.check(docstring):
                bad_results.add(bad_statement)
        return bad_results

    def run(self, api_name: str, docstring: str) -> typing.List[TestResult]:
        if False:
            while True:
                i = 10
        'Run the xdoctest with a docstring.'
        bad_results = self._check_bad_statements(docstring)
        if bad_results:
            for bad_statement in bad_results:
                logger.warning('%s >>> %s', api_name, bad_statement.msg)
            return [TestResult(name=api_name, badstatement=True)]
        (docstring, directives) = self._parse_directive(docstring)
        (examples_to_test, examples_nocode) = self._extract_examples(api_name, docstring, **directives)
        try:
            result = self._execute_xdoctest(examples_to_test, examples_nocode, **directives)
        except queue.Empty:
            result = [TestResult(name=api_name, timeout=True, time=directives.get('timeout', TEST_TIMEOUT))]
        return result

    def _extract_examples(self, api_name, docstring, **directives):
        if False:
            i = 10
            return i + 15
        'Extract code block examples from docstring.'
        examples_to_test = {}
        examples_nocode = {}
        for (example_idx, example) in enumerate(self.docstring_parser(docstr=docstring, callname=api_name)):
            example.mode = self.mode
            example.config.update(self.config)
            example_key = f'{api_name}_{example_idx}'
            if not example._parts:
                examples_nocode[example_key] = example
                continue
            examples_to_test[example_key] = example
        if not examples_nocode and (not examples_to_test):
            examples_nocode[api_name] = api_name
        return (examples_to_test, examples_nocode)

    def _execute_xdoctest(self, examples_to_test, examples_nocode, **directives):
        if False:
            for i in range(10):
                print('nop')
        if directives.get('solo') is not None:
            return self._execute(examples_to_test, examples_nocode)
        if self._use_multiprocessing:
            _ctx = multiprocessing.get_context('spawn')
            result_queue = _ctx.Queue()
            exec_processer = functools.partial(_ctx.Process, daemon=True)
        else:
            result_queue = queue.Queue()
            exec_processer = functools.partial(threading.Thread, daemon=True)
        processer = exec_processer(target=self._execute_with_queue, args=(result_queue, examples_to_test, examples_nocode))
        processer.start()
        result = result_queue.get(timeout=directives.get('timeout', TEST_TIMEOUT))
        processer.join()
        return result

    def _execute(self, examples_to_test, examples_nocode):
        if False:
            i = 10
            return i + 15
        'Run xdoctest for each example'
        self._patch_xdoctest()
        test_results = []
        for (_, example) in examples_to_test.items():
            start_time = time.time()
            result = example.run(verbose=self.verbose, on_error='return')
            end_time = time.time()
            test_results.append(TestResult(name=str(example), passed=result['passed'], skipped=result['skipped'], failed=result['failed'], test_msg=str(result['exc_info']), time=end_time - start_time))
        for (_, example) in examples_nocode.items():
            test_results.append(TestResult(name=str(example), nocode=True))
        return test_results

    def _execute_with_queue(self, queue, examples_to_test, examples_nocode):
        if False:
            while True:
                i = 10
        queue.put(self._execute(examples_to_test, examples_nocode))

    def print_summary(self, test_results, whl_error=None):
        if False:
            for i in range(10):
                print('nop')
        summary = collections.defaultdict(list)
        is_fail = False
        logger.warning('----------------Check results--------------------')
        logger.warning('>>> Sample code test capacity: %s', self._test_capacity)
        if whl_error is not None and whl_error:
            logger.warning('%s is not in whl.', whl_error)
            logger.warning('')
            logger.warning('Please check the whl package and API_PR.spec!')
            logger.warning('You can follow these steps in order to generate API.spec:')
            logger.warning('1. cd ${paddle_path}, compile paddle;')
            logger.warning('2. pip install build/python/dist/(build whl package);')
            logger.warning("3. run 'python tools/print_signatures.py paddle > paddle/fluid/API.spec'.")
            for test_result in test_results:
                if test_result.failed:
                    logger.error('In addition, mistakes found in sample codes: %s', test_result.name)
            log_exit(1)
        else:
            for test_result in test_results:
                summary[test_result.state].append(test_result)
                if test_result.state.is_fail:
                    is_fail = True
            summary = sorted(summary.items(), key=lambda x: x[0].order)
            for (result_cls, result_list) in summary:
                logging_msg = result_cls.msg(len(result_list), self._test_capacity)
                result_cls.logger(logging_msg)
                result_cls.logger('\n'.join([str(r) for r in result_list]))
            if is_fail:
                logger.warning('>>> Mistakes found in sample codes in env: %s!', self._test_capacity)
                logger.warning('>>> Please recheck the sample codes.')
                log_exit(1)
        logger.warning('>>> Sample code check is successful in env: %s!', self._test_capacity)
        logger.warning('----------------End of the Check--------------------')
if __name__ == '__main__':
    args = parse_args()
    run_doctest(args, doctester=Xdoctester(debug=args.debug))