"""Sphinx documentation plugin used to document tasks.

Introduction
============

Usage
-----

The Celery extension for Sphinx requires Sphinx 2.0 or later.

Add the extension to your :file:`docs/conf.py` configuration module:

.. code-block:: python

    extensions = (...,
                  'celery.contrib.sphinx')

If you'd like to change the prefix for tasks in reference documentation
then you can change the ``celery_task_prefix`` configuration value:

.. code-block:: python

    celery_task_prefix = '(task)'  # < default

With the extension installed `autodoc` will automatically find
task decorated objects (e.g. when using the automodule directive)
and generate the correct (as well as add a ``(task)`` prefix),
and you can also refer to the tasks using `:task:proj.tasks.add`
syntax.

Use ``.. autotask::`` to alternatively manually document a task.
"""
from inspect import signature
from docutils import nodes
from sphinx.domains.python import PyFunction
from sphinx.ext.autodoc import FunctionDocumenter
from celery.app.task import BaseTask

class TaskDocumenter(FunctionDocumenter):
    """Document task definitions."""
    objtype = 'task'
    member_order = 11

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        if False:
            while True:
                i = 10
        return isinstance(member, BaseTask) and getattr(member, '__wrapped__')

    def format_args(self):
        if False:
            for i in range(10):
                print('nop')
        wrapped = getattr(self.object, '__wrapped__', None)
        if wrapped is not None:
            sig = signature(wrapped)
            if 'self' in sig.parameters or 'cls' in sig.parameters:
                sig = sig.replace(parameters=list(sig.parameters.values())[1:])
            return str(sig)
        return ''

    def document_members(self, all_members=False):
        if False:
            return 10
        pass

    def check_module(self):
        if False:
            print('Hello World!')
        wrapped = getattr(self.object, '__wrapped__', None)
        if wrapped and getattr(wrapped, '__module__') == self.modname:
            return True
        return super().check_module()

class TaskDirective(PyFunction):
    """Sphinx task directive."""

    def get_signature_prefix(self, sig):
        if False:
            for i in range(10):
                print('nop')
        return [nodes.Text(self.env.config.celery_task_prefix)]

def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    if False:
        for i in range(10):
            print('nop')
    'Handler for autodoc-skip-member event.'
    if isinstance(obj, BaseTask) and getattr(obj, '__wrapped__'):
        if skip:
            return False
    return None

def setup(app):
    if False:
        while True:
            i = 10
    'Setup Sphinx extension.'
    app.setup_extension('sphinx.ext.autodoc')
    app.add_autodocumenter(TaskDocumenter)
    app.add_directive_to_domain('py', 'task', TaskDirective)
    app.add_config_value('celery_task_prefix', '(task)', True)
    app.connect('autodoc-skip-member', autodoc_skip_member_handler)
    return {'parallel_read_safe': True}