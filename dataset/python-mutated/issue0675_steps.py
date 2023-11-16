from __future__ import absolute_import, print_function
import os.path
from behave import given, when, then
import six

@given(u'I create a symlink from "{source}" to "{dest}"')
@when(u'I create a symlink from "{source}" to "{dest}"')
def step_create_symlink(context, source, dest):
    if False:
        for i in range(10):
            print('nop')
    print('symlink: %s -> %s' % (source, dest))
    text = u'When I run "ln -s {source} {dest}"'.format(source=source, dest=dest)
    context.execute_steps(text)
    if False:
        source_is_dir = os.path.isdir(source)
        if six.py3 and source_is_dir:
            os.symlink(source, dest, target_is_directory=source_is_dir)
        else:
            os.symlink(source, dest)