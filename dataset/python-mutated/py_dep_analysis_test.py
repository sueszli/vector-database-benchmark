import os
import tempfile
import unittest
import py_dep_analysis as pda

class TestPyDepAnalysis(unittest.TestCase):

    def create_tmp_file(self, path: str, content: str):
        if False:
            return 10
        with open(path, 'w') as f:
            f.write(content)

    def test_full_module_path(self):
        if False:
            return 10
        self.assertEqual(pda._full_module_path('aa.bb.cc', '__init__.py'), 'aa.bb.cc')
        self.assertEqual(pda._full_module_path('aa.bb.cc', 'dd.py'), 'aa.bb.cc.dd')
        self.assertEqual(pda._full_module_path('', 'dd.py'), 'dd')

    def test_bazel_path_to_module_path(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(pda._bazel_path_to_module_path('//python/ray/rllib:xxx/yyy/dd'), 'ray.rllib.xxx.yyy.dd')
        self.assertEqual(pda._bazel_path_to_module_path('python:ray/rllib/xxx/yyy/dd'), 'ray.rllib.xxx.yyy.dd')
        self.assertEqual(pda._bazel_path_to_module_path('python/ray/rllib:xxx/yyy/dd'), 'ray.rllib.xxx.yyy.dd')

    def test_file_path_to_module_path(self):
        if False:
            return 10
        self.assertEqual(pda._file_path_to_module_path('python/ray/rllib/env/env.py'), 'ray.rllib.env.env')
        self.assertEqual(pda._file_path_to_module_path('python/ray/rllib/env/__init__.py'), 'ray.rllib.env')

    def test_import_line_continuation(self):
        if False:
            return 10
        graph = pda.DepGraph()
        graph.ids['ray'] = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, 'continuation1.py')
            self.create_tmp_file(src_path, '\nimport ray.rllib.env.\\\n    mock_env\nb = 2\n')
            pda._process_file(graph, src_path, 'ray')
        self.assertEqual(len(graph.ids), 2)
        print(graph.ids)
        self.assertEqual(graph.ids['ray.rllib.env.mock_env'], 1)
        self.assertEqual(graph.edges[0], {1: True})

    def test_import_line_continuation_parenthesis(self):
        if False:
            i = 10
            return i + 15
        graph = pda.DepGraph()
        graph.ids['ray'] = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, 'continuation1.py')
            self.create_tmp_file(src_path, '\nfrom ray.rllib.env import (ClassName,\n    module1, module2)\nb = 2\n')
            pda._process_file(graph, src_path, 'ray')
        self.assertEqual(len(graph.ids), 2)
        print(graph.ids)
        self.assertEqual(graph.ids['ray.rllib.env'], 1)
        self.assertEqual(graph.edges[0], {1: True})

    def test_from_import_file_module(self):
        if False:
            i = 10
            return i + 15
        graph = pda.DepGraph()
        graph.ids['ray'] = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = 'multi_line_comment_3.py'
            self.create_tmp_file(os.path.join(tmpdir, src_path), '\nfrom ray.rllib.env import mock_env\na = 1\nb = 2\n')
            module_dir = os.path.join(tmpdir, 'python', 'ray', 'rllib', 'env')
            os.makedirs(module_dir, exist_ok=True)
            f = open(os.path.join(module_dir, 'mock_env.py'), 'w')
            f.write("print('hello world!')")
            f.close
            pda._process_file(graph, src_path, 'ray', _base_dir=tmpdir)
        self.assertEqual(len(graph.ids), 2)
        self.assertEqual(graph.ids['ray.rllib.env.mock_env'], 1)
        self.assertEqual(graph.edges[0], {1: True})

    def test_from_import_class_object(self):
        if False:
            for i in range(10):
                print('nop')
        graph = pda.DepGraph()
        graph.ids['ray'] = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = 'multi_line_comment_3.py'
            self.create_tmp_file(os.path.join(tmpdir, src_path), '\nfrom ray.rllib.env import MockEnv\na = 1\nb = 2\n')
            module_dir = os.path.join(tmpdir, 'python', 'ray', 'rllib')
            os.makedirs(module_dir, exist_ok=True)
            f = open(os.path.join(module_dir, 'env.py'), 'w')
            f.write("print('hello world!')")
            f.close
            pda._process_file(graph, src_path, 'ray', _base_dir=tmpdir)
        self.assertEqual(len(graph.ids), 2)
        self.assertEqual(graph.ids['ray.rllib.env'], 1)
        self.assertEqual(graph.edges[0], {1: True})
if __name__ == '__main__':
    import pytest
    import sys
    sys.exit(pytest.main(['-v', __file__]))