"""
Provides :mod:`behave` steps to provide and use "working directory"
as base directory to:

* Create files
* Create directories
"""
from __future__ import absolute_import, print_function
import os
import shutil
from behave import given, step
from behave4cmd0 import command_util

@given(u'a new working directory')
def step_a_new_working_directory(context):
    if False:
        i = 10
        return i + 15
    'Creates a new, empty working directory.'
    command_util.ensure_context_attribute_exists(context, 'workdir', None)
    command_util.ensure_workdir_exists(context)
    shutil.rmtree(context.workdir, ignore_errors=True)
    command_util.ensure_workdir_exists(context)

@given(u'I use the current directory as working directory')
def step_use_curdir_as_working_directory(context):
    if False:
        i = 10
        return i + 15
    'Uses the current directory as working directory'
    context.workdir = os.path.abspath('.')
    command_util.ensure_workdir_exists(context)

@step(u'I use the directory "{directory}" as working directory')
def step_use_directory_as_working_directory(context, directory):
    if False:
        i = 10
        return i + 15
    'Uses the directory as new working directory'
    command_util.ensure_context_attribute_exists(context, 'workdir', None)
    current_workdir = context.workdir
    if not current_workdir:
        current_workdir = os.getcwd()
    if not os.path.isabs(directory):
        new_workdir = os.path.join(current_workdir, directory)
        exists_relto_current_dir = os.path.isdir(directory)
        exists_relto_current_workdir = os.path.isdir(new_workdir)
        if exists_relto_current_workdir or not exists_relto_current_dir:
            workdir = new_workdir
        else:
            assert exists_relto_current_workdir
            workdir = directory
        workdir = os.path.abspath(workdir)
    context.workdir = workdir
    command_util.ensure_workdir_exists(context)