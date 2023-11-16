"""Ensure the argparse parser and Options class are in sync.

In particular, verify that the argparse defaults are the same as the Options
defaults, and that argparse doesn't assign any new members to the Options
object it creates.
"""
from __future__ import annotations
import argparse
import sys
from mypy.main import infer_python_executable, process_options
from mypy.options import Options
from mypy.test.helpers import Suite, assert_equal

class ArgSuite(Suite):

    def test_coherence(self) -> None:
        if False:
            i = 10
            return i + 15
        options = Options()
        (_, parsed_options) = process_options([], require_targets=False)
        options.config_file = parsed_options.config_file
        assert_equal(options.snapshot(), parsed_options.snapshot())

    def test_executable_inference(self) -> None:
        if False:
            while True:
                i = 10
        'Test the --python-executable flag with --python-version'
        sys_ver_str = '{ver.major}.{ver.minor}'.format(ver=sys.version_info)
        base = ['file.py']
        matching_version = base + [f'--python-version={sys_ver_str}']
        (_, options) = process_options(matching_version)
        assert options.python_version == sys.version_info[:2]
        assert options.python_executable == sys.executable
        matching_version = base + [f'--python-executable={sys.executable}']
        (_, options) = process_options(matching_version)
        assert options.python_version == sys.version_info[:2]
        assert options.python_executable == sys.executable
        matching_version = base + [f'--python-version={sys_ver_str}', f'--python-executable={sys.executable}']
        (_, options) = process_options(matching_version)
        assert options.python_version == sys.version_info[:2]
        assert options.python_executable == sys.executable
        matching_version = base + [f'--python-version={sys_ver_str}', '--no-site-packages']
        (_, options) = process_options(matching_version)
        assert options.python_version == sys.version_info[:2]
        assert options.python_executable is None
        special_opts = argparse.Namespace()
        special_opts.python_executable = None
        special_opts.python_version = None
        special_opts.no_executable = None
        options = Options()
        options.python_executable = None
        options.python_version = sys.version_info[:2]
        infer_python_executable(options, special_opts)
        assert options.python_version == sys.version_info[:2]
        assert options.python_executable == sys.executable
        options = Options()
        options.python_executable = sys.executable
        infer_python_executable(options, special_opts)
        assert options.python_version == sys.version_info[:2]
        assert options.python_executable == sys.executable