"""
Invoke build script (python based).

.. seealso:: https://github.com/pyinvoke/invoke
"""
from __future__ import print_function
import sys
from invoke import task, Collection
USE_PTY = sys.stdout.isatty()

@task(help={'args': 'Command line args for behave', 'format': 'Formatter to use'})
def behave_test(ctx, args='', format=''):
    if False:
        for i in range(10):
            print('nop')
    'Run behave tests.'
    format = format or ctx.behave_test.format
    options = ctx.behave_test.options or ''
    args = args or ctx.behave_test.args
    behave = '{python} bin/behave'.format(python=sys.executable)
    ctx.run('{behave} -f {format} {options} {args}'.format(behave=behave, format=format, options=options, args=args), pty=USE_PTY)
namespace = Collection()
namespace.add_task(behave_test, default=True)
namespace.configure({'behave_test': {'args': '', 'format': 'progress2', 'options': ''}})