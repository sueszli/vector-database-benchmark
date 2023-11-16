"""Utilities for using the kubernetes decorator."""
from __future__ import annotations
import os
from collections import deque
import jinja2
from jinja2 import select_autoescape

def _balance_parens(after_decorator):
    if False:
        return 10
    num_paren = 1
    after_decorator = deque(after_decorator)
    after_decorator.popleft()
    while num_paren:
        current = after_decorator.popleft()
        if current == '(':
            num_paren += 1
        elif current == ')':
            num_paren -= 1
    return ''.join(after_decorator)

def remove_task_decorator(python_source: str, task_decorator_name: str) -> str:
    if False:
        return 10
    '\n    Remove @task.kubernetes or similar as well as @setup and @teardown.\n\n    :param python_source: python source code\n    :param task_decorator_name: the task decorator name\n    '

    def _remove_task_decorator(py_source, decorator_name):
        if False:
            for i in range(10):
                print('nop')
        if decorator_name not in py_source:
            return python_source
        split = python_source.split(decorator_name)
        (before_decorator, after_decorator) = (split[0], split[1])
        if after_decorator[0] == '(':
            after_decorator = _balance_parens(after_decorator)
        if after_decorator[0] == '\n':
            after_decorator = after_decorator[1:]
        return before_decorator + after_decorator
    decorators = ['@setup', '@teardown', task_decorator_name]
    for decorator in decorators:
        python_source = _remove_task_decorator(python_source, decorator)
    return python_source

def write_python_script(jinja_context: dict, filename: str, render_template_as_native_obj: bool=False):
    if False:
        while True:
            i = 10
    '\n    Render the python script to a file to execute in the virtual environment.\n\n    :param jinja_context: The jinja context variables to unpack and replace with its placeholders in the\n        template file.\n    :param filename: The name of the file to dump the rendered script to.\n    :param render_template_as_native_obj: If ``True``, rendered Jinja template would be converted\n        to a native Python object\n    '
    template_loader = jinja2.FileSystemLoader(searchpath=os.path.dirname(__file__))
    template_env: jinja2.Environment
    if render_template_as_native_obj:
        template_env = jinja2.nativetypes.NativeEnvironment(loader=template_loader, undefined=jinja2.StrictUndefined)
    else:
        template_env = jinja2.Environment(loader=template_loader, undefined=jinja2.StrictUndefined, autoescape=select_autoescape(['html', 'xml']))
    template = template_env.get_template('python_kubernetes_script.jinja2')
    template.stream(**jinja_context).dump(filename)