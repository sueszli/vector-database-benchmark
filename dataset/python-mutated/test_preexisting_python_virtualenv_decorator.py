from __future__ import annotations
from airflow.utils.decorators import remove_task_decorator

class TestExternalPythonDecorator:

    def test_remove_task_decorator(self):
        if False:
            i = 10
            return i + 15
        py_source = '@task.external_python(use_dill=True)\ndef f():\nimport funcsigs'
        res = remove_task_decorator(python_source=py_source, task_decorator_name='@task.external_python')
        assert res == 'def f():\nimport funcsigs'

    def test_remove_decorator_no_parens(self):
        if False:
            return 10
        py_source = '@task.external_python\ndef f():\nimport funcsigs'
        res = remove_task_decorator(python_source=py_source, task_decorator_name='@task.external_python')
        assert res == 'def f():\nimport funcsigs'

    def test_remove_decorator_nested(self):
        if False:
            while True:
                i = 10
        py_source = '@foo\n@task.external_python\n@bar\ndef f():\nimport funcsigs'
        res = remove_task_decorator(python_source=py_source, task_decorator_name='@task.external_python')
        assert res == '@foo\n@bar\ndef f():\nimport funcsigs'
        py_source = '@foo\n@task.external_python()\n@bar\ndef f():\nimport funcsigs'
        res = remove_task_decorator(python_source=py_source, task_decorator_name='@task.external_python')
        assert res == '@foo\n@bar\ndef f():\nimport funcsigs'