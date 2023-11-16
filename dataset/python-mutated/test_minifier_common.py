import dataclasses
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Optional
from unittest.mock import patch
import torch
import torch._dynamo
import torch._dynamo.test_case
from torch.utils._traceback import report_compile_source_on_error

@dataclasses.dataclass
class MinifierTestResult:
    minifier_code: str
    repro_code: str

    def _get_module(self, t):
        if False:
            for i in range(10):
                print('nop')
        match = re.search('class Repro\\(torch\\.nn\\.Module\\):\\s+([ ].*\\n| *\\n)+', t)
        assert match is not None, 'failed to find module'
        r = match.group(0)
        r = re.sub('\\s+$', '\n', r, flags=re.MULTILINE)
        r = re.sub('\\n{3,}', '\n\n', r)
        return r.strip()

    def minifier_module(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_module(self.minifier_code)

    def repro_module(self):
        if False:
            i = 10
            return i + 15
        return self._get_module(self.repro_code)

class MinifierTestBase(torch._dynamo.test_case.TestCase):
    DEBUG_DIR = tempfile.mkdtemp()

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        cls._exit_stack.enter_context(torch._dynamo.config.patch(debug_dir_root=cls.DEBUG_DIR))
        cls._exit_stack.enter_context(torch._inductor.config.patch({'pattern_matcher': False, 'compile_threads': 1, 'cpp.vec_isa_ok': False}))

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        if os.getenv('PYTORCH_KEEP_TMPDIR', '0') != '1':
            shutil.rmtree(cls.DEBUG_DIR)
        else:
            print(f'test_minifier_common tmpdir kept at: {cls.DEBUG_DIR}')
        cls._exit_stack.close()

    def _gen_codegen_fn_patch_code(self, device, bug_type):
        if False:
            i = 10
            return i + 15
        assert bug_type in ('compile_error', 'runtime_error', 'accuracy')
        return f"{torch._dynamo.config.codegen_config()}\n{torch._inductor.config.codegen_config()}\ntorch._inductor.config.{('cpp' if device == 'cpu' else 'triton')}.inject_relu_bug_TESTING_ONLY = {bug_type!r}\n"

    def _maybe_subprocess_run(self, args, *, isolate, cwd=None):
        if False:
            i = 10
            return i + 15
        if not isolate:
            assert len(args) >= 2, args
            assert args[0] == 'python3', args
            if args[1] == '-c':
                assert len(args) == 3, args
                code = args[2]
                args = ['-c']
            else:
                assert len(args) >= 2, args
                with open(args[1]) as f:
                    code = f.read()
                args = args[1:]
            dynamo_config = torch._dynamo.config.shallow_copy_dict()
            inductor_config = torch._inductor.config.shallow_copy_dict()
            try:
                stderr = io.StringIO()
                log_handler = logging.StreamHandler(stderr)
                log = logging.getLogger('torch._dynamo')
                log.addHandler(log_handler)
                try:
                    prev_cwd = os.getcwd()
                    if cwd is not None:
                        os.chdir(cwd)
                    with patch('sys.argv', args), report_compile_source_on_error():
                        exec(code, {'__name__': '__main__', '__compile_source__': code})
                    rc = 0
                except Exception:
                    rc = 1
                    traceback.print_exc(file=stderr)
                finally:
                    log.removeHandler(log_handler)
                    if cwd is not None:
                        os.chdir(prev_cwd)
                    torch._dynamo.reset()
            finally:
                torch._dynamo.config.load_config(dynamo_config)
                torch._inductor.config.load_config(inductor_config)
            return subprocess.CompletedProcess(args, rc, b'', stderr.getvalue().encode('utf-8'))
        else:
            return subprocess.run(args, capture_output=True, cwd=cwd, check=False)

    def _run_test_code(self, code, *, isolate):
        if False:
            i = 10
            return i + 15
        proc = self._maybe_subprocess_run(['python3', '-c', code], isolate=isolate, cwd=self.DEBUG_DIR)
        print('test stdout:', proc.stdout.decode('utf-8'))
        print('test stderr:', proc.stderr.decode('utf-8'))
        repro_dir_match = re.search('(\\S+)minifier_launcher.py', proc.stderr.decode('utf-8'))
        if repro_dir_match is not None:
            return (proc, repro_dir_match.group(1))
        return (proc, None)

    def _run_minifier_launcher(self, repro_dir, isolate, *, minifier_args=()):
        if False:
            return 10
        self.assertIsNotNone(repro_dir)
        launch_file = os.path.join(repro_dir, 'minifier_launcher.py')
        with open(launch_file) as f:
            launch_code = f.read()
        self.assertTrue(os.path.exists(launch_file))
        args = ['python3', launch_file, 'minify', *minifier_args]
        if not isolate:
            args.append('--no-isolate')
        launch_proc = self._maybe_subprocess_run(args, isolate=isolate, cwd=repro_dir)
        print('minifier stdout:', launch_proc.stdout.decode('utf-8'))
        stderr = launch_proc.stderr.decode('utf-8')
        print('minifier stderr:', stderr)
        self.assertNotIn('Input graph did not fail the tester', stderr)
        return (launch_proc, launch_code)

    def _run_repro(self, repro_dir, *, isolate=True):
        if False:
            return 10
        self.assertIsNotNone(repro_dir)
        repro_file = os.path.join(repro_dir, 'repro.py')
        with open(repro_file) as f:
            repro_code = f.read()
        self.assertTrue(os.path.exists(repro_file))
        repro_proc = self._maybe_subprocess_run(['python3', repro_file], isolate=isolate, cwd=repro_dir)
        print('repro stdout:', repro_proc.stdout.decode('utf-8'))
        print('repro stderr:', repro_proc.stderr.decode('utf-8'))
        return (repro_proc, repro_code)

    def _gen_test_code(self, run_code, repro_after, repro_level):
        if False:
            print('Hello World!')
        return f'import torch\nimport torch._dynamo\n{torch._dynamo.config.codegen_config()}\n{torch._inductor.config.codegen_config()}\ntorch._dynamo.config.repro_after = "{repro_after}"\ntorch._dynamo.config.repro_level = {repro_level}\ntorch._dynamo.config.debug_dir_root = "{self.DEBUG_DIR}"\n{run_code}\n'

    def _run_full_test(self, run_code, repro_after, expected_error, *, isolate, minifier_args=()) -> Optional[MinifierTestResult]:
        if False:
            while True:
                i = 10
        if isolate:
            repro_level = 3
        elif expected_error is None or expected_error == 'AccuracyError':
            repro_level = 4
        else:
            repro_level = 2
        test_code = self._gen_test_code(run_code, repro_after, repro_level)
        print('running test', file=sys.stderr)
        (test_proc, repro_dir) = self._run_test_code(test_code, isolate=isolate)
        if expected_error is None:
            self.assertEqual(test_proc.returncode, 0)
            self.assertIsNone(repro_dir)
            return None
        self.assertIn(expected_error, test_proc.stderr.decode('utf-8'))
        self.assertIsNotNone(repro_dir)
        print('running minifier', file=sys.stderr)
        (minifier_proc, minifier_code) = self._run_minifier_launcher(repro_dir, isolate=isolate, minifier_args=minifier_args)
        print('running repro', file=sys.stderr)
        (repro_proc, repro_code) = self._run_repro(repro_dir, isolate=isolate)
        self.assertIn(expected_error, repro_proc.stderr.decode('utf-8'))
        self.assertNotEqual(repro_proc.returncode, 0)
        return MinifierTestResult(minifier_code=minifier_code, repro_code=repro_code)